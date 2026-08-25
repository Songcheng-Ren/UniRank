# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from .QFormerCross31 import (
    FieldAwareSDPACrossValueAttention,
    count_non_sequence_fields,
    scale_normalized_field_keys,
)
from .QFormerCross8 import (
    FeedForward,
    QFormerCross8,
    SDPAMultiHeadAttention,
)


class FieldAwareMaskedSDPAMultiHeadAttention(nn.Module):
    """Standard SDPA read with static field/head logit calibration."""

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
        self.w_v = nn.Linear(token_dim, token_dim)
        self.w_o = nn.Linear(token_dim, token_dim)
        self.q_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )

    def forward(self, query, key, value, field_head_log_scale, mask=None):
        batch_size, query_length, _ = query.shape
        num_fields = key.size(1)
        q = self.w_q(query).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.w_k(key).view(
            batch_size, num_fields, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.w_v(value).view(
            batch_size, num_fields, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        k = scale_normalized_field_keys(k, field_head_log_scale)

        key_mask = None
        has_valid_key = None
        if mask is not None:
            valid = mask.to(dtype=torch.bool, device=q.device)
            key_mask = valid.view(batch_size, 1, 1, num_fields)
            has_valid_key = valid.any(dim=1).view(batch_size, 1, 1)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_length, self.token_dim
        )
        output = self.w_o(output)
        if has_valid_key is not None:
            output = output * has_valid_key.to(dtype=output.dtype)
        return output


class FieldAwareRecursiveCrossValueLayer(nn.Module):
    """V8 recurrence using the stage-level shared field/head scale."""

    def __init__(self, token_dim, num_heads, dropout=0.0, qk_norm=False):
        super().__init__()
        self.cross_attn = FieldAwareSDPACrossValueAttention(
            token_dim, num_heads, qk_norm
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, features, field_head_log_scale, mask=None):
        cross_output = self.cross_attn(
            queries,
            features,
            features,
            field_head_log_scale,
            mask,
        )
        return queries + self.dropout(cross_output)


class FieldAwareRecursiveCrossValueStage(nn.Module):
    """V8 NS stage sharing one alpha across initial and recursive reads."""

    def __init__(self, token_dim, num_heads, num_layers, num_queries, ffn_dim,
                 num_fields, dropout=0.0, qk_norm=False):
        super().__init__()
        self.token_dim = token_dim
        self.num_queries = num_queries
        self.learnable_queries = nn.Parameter(
            torch.empty(num_queries, token_dim)
        )
        self.field_head_log_scale = nn.Parameter(
            torch.zeros(num_fields, num_heads)
        )
        self.initial_read = FieldAwareMaskedSDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.initial_norm = nn.LayerNorm(token_dim)
        self.layers = nn.ModuleList([
            FieldAwareRecursiveCrossValueLayer(
                token_dim=token_dim,
                num_heads=num_heads,
                dropout=dropout,
                qk_norm=qk_norm,
            )
            for _ in range(num_layers)
        ])
        self.post_self_attn = SDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.post_self_norm = nn.LayerNorm(token_dim)
        self.post_ffn = FeedForward(token_dim, ffn_dim, dropout)
        self.post_ffn_norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.learnable_queries)
        nn.init.zeros_(self.field_head_log_scale)

    def forward(self, features, mask=None):
        batch_size = features.size(0)
        queries = self.learnable_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        initial_delta = self.initial_read(
            queries,
            features,
            features,
            self.field_head_log_scale,
            mask,
        )
        queries = self.initial_norm(
            queries + self.dropout(initial_delta)
        )
        for layer in self.layers:
            queries = layer(
                queries,
                features,
                self.field_head_log_scale,
                mask,
            )

        mixed = self.post_self_attn(queries, queries, queries)
        queries = self.post_self_norm(queries + self.dropout(mixed))
        ffn_output = self.post_ffn(queries)
        return self.post_ffn_norm(
            queries + self.dropout(ffn_output)
        )


class QFormerCross81(QFormerCross8):
    """QFormerCross8 plus shared field/head calibration in the NS stage."""

    def __init__(self, feature_map, model_id="QFormerCross81", **kwargs):
        num_fields = count_non_sequence_fields(feature_map)
        ns_stage_cls = partial(
            FieldAwareRecursiveCrossValueStage,
            num_fields=num_fields,
        )
        super().__init__(
            feature_map,
            model_id=model_id,
            _ns_qformer_cls=ns_stage_cls,
            **kwargs,
        )
        self.ns_qformer.field_names = tuple(self.ns_features)

