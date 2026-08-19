# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

import math

import torch
from torch import nn
import torch.nn.functional as F

from .QFormerCross import FeedForward
from .QFormerCross2 import QFormerCross2


class QFormerCross15(QFormerCross2):
    """QFormerCross3 architecture with RMS-normalized attention Q/K.

    The field tokenization, non-sequential-first data flow, positional sequence
    representation and prediction towers are intentionally kept identical to
    QFormerCross2. Only the QFormer attention primitives change:

    * self-attention uses PyTorch scaled_dot_product_attention;
    * attention Q/K heads use RMSNorm instead of LayerNorm;
    * CrossValue uses standard head-local Q/K/V layouts, performs SDPA first,
      and then explicitly crosses each query with its attended feature value.

    This isolates the throughput-oriented attention change as a separate model
    so QFormerCross2 remains a stable accuracy baseline.
    """

    def __init__(self, feature_map, model_id="QFormerCross15", **kwargs):
        super().__init__(
            feature_map,
            model_id=model_id,
            _qformer_stage_cls=SDPAQFormerStage,
            _qformer_layer_cls=SDPAQFormerLayer,
            **kwargs,
        )


class SDPAQFormerStage(nn.Module):
    """Input-conditioned learned queries backed by SDPA QFormer layers."""

    def __init__(self, token_dim, num_heads, num_layers, num_queries, ffn_dim,
                 dropout=0.0, qk_norm=False):
        super().__init__()
        self.token_dim = token_dim
        self.num_queries = num_queries
        self.learnable_queries = nn.Parameter(torch.empty(num_queries, token_dim))
        self.context_proj = nn.Linear(token_dim, token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.query_norm = nn.LayerNorm(token_dim)
        self.layers = nn.ModuleList([
            SDPAQFormerLayer(
                token_dim=token_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                qk_norm=qk_norm,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(token_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.learnable_queries)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

    def forward(self, features, mask=None):
        batch_size = features.size(0)
        if mask is None:
            pooled_features = features.mean(dim=1)
        else:
            valid = mask.to(
                dtype=features.dtype, device=features.device
            ).unsqueeze(-1)
            pooled_features = (features * valid).sum(dim=1)
            pooled_features = pooled_features / valid.sum(dim=1).clamp_min(1.0)

        context = self.context_norm(self.context_proj(pooled_features))
        queries = self.learnable_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        queries = self.query_norm(queries + context.unsqueeze(1))
        for layer in self.layers:
            queries = layer(queries, features, mask)
        return self.final_norm(queries)


class SDPAQFormerLayer(nn.Module):
    def __init__(self, token_dim, num_heads, ffn_dim, dropout=0.0,
                 qk_norm=False):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(token_dim)
        self.self_attn = SDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.cross_norm = nn.LayerNorm(token_dim)
        self.cross_attn = SDPACrossValueAttention(
            token_dim, num_heads, qk_norm
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = FeedForward(token_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, features, mask=None):
        attn_out = self.self_attn(queries, queries, queries)
        queries = self.self_attn_norm(queries + self.dropout(attn_out))
        cross_out = self.cross_attn(queries, features, features, mask)
        queries = self.cross_norm(queries + self.dropout(cross_out))
        ffn_out = self.ffn(queries)
        return self.ffn_norm(queries + self.dropout(ffn_out))


class SDPAMultiHeadAttention(nn.Module):
    def __init__(self, token_dim, num_heads, qk_norm=False):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.w_q = nn.Linear(token_dim, token_dim)
        self.w_k = nn.Linear(token_dim, token_dim)
        self.w_v = nn.Linear(token_dim, token_dim)
        self.w_o = nn.Linear(token_dim, token_dim)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, query, key, value):
        batch_size, query_length, _ = query.shape
        key_length = key.size(1)
        q = self.w_q(query).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.w_k(key).view(
            batch_size, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.w_v(value).view(
            batch_size, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_length, -1
        )
        return self.w_o(output)


class SDPACrossValueAttention(nn.Module):
    """Head-local SDPA followed by an explicit query-feature product."""

    def __init__(self, token_dim, num_heads, qk_norm=False):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.w_q = nn.Linear(token_dim, token_dim)
        self.w_k = nn.Linear(token_dim, token_dim)
        self.w_qi = nn.Linear(token_dim, token_dim)
        self.w_fi = nn.Linear(token_dim, token_dim)
        self.w_v_pair = nn.Linear(token_dim, token_dim)
        self.w_o = nn.Linear(token_dim, token_dim)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, queries, keys, values, mask=None):
        batch_size, num_queries, _ = queries.shape
        sequence_length = keys.size(1)
        q = self.w_q(queries).view(
            batch_size, num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.w_k(keys).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.w_fi(values).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        key_mask = None
        if mask is not None:
            key_mask = mask.to(dtype=torch.bool, device=q.device).view(
                batch_size, 1, 1, sequence_length
            )
        attended_features = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )

        q_interaction = self.w_qi(queries).view(
            batch_size, num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        pair_summary = q_interaction * attended_features
        pair_summary = pair_summary.transpose(1, 2).contiguous().view(
            batch_size, num_queries, self.token_dim
        )
        return self.w_o(self.w_v_pair(pair_summary))
