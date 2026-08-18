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
from .QFormerCross3 import (
    SDPACrossValueAttention,
    SDPAMultiHeadAttention,
)


class QFormerCross7(QFormerCross2):
    """QFormerCross3 backbone with an order-growing NS cross recurrence.

    The non-sequential stage first reads the raw field tokens into a fixed
    number of latent queries.  It then repeatedly applies the same structural
    recurrence against the unchanged first-order field tokens::

        Q_(l+1) = Q_l + CrossValue(Q_l, X)

    Since CrossValue is bilinear in its query/value path, each recurrence can
    introduce one additional field-interaction order while the residual keeps
    all lower-order terms.  Query mixing and the FFN are applied once after the
    complete cross stack, so they do not obscure the layer/order correspondence.

    CrossValue is intentionally restricted to the non-sequential encoder.  The
    second stage uses ordinary SDPA cross-attention to extract sequential
    interests with the already-crossed queries.
    """

    def __init__(self, feature_map, model_id="QFormerCross7", **kwargs):
        super().__init__(
            feature_map,
            model_id=model_id,
            _qformer_stage_cls=RecursiveCrossValueStage,
            _qformer_layer_cls=SDPASequenceQFormerLayer,
            **kwargs,
        )


class RecursiveCrossValueStage(nn.Module):
    """Read fields once, recursively cross with raw fields, then mix queries."""

    def __init__(self, token_dim, num_heads, num_layers, num_queries, ffn_dim,
                 dropout=0.0, qk_norm=False):
        super().__init__()
        self.token_dim = token_dim
        self.num_queries = num_queries
        self.learnable_queries = nn.Parameter(
            torch.empty(num_queries, token_dim)
        )

        # Q0 is a first-order, input-conditioned representation of the raw
        # non-sequential field tokens.
        self.initial_read = MaskedSDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.initial_norm = nn.LayerNorm(token_dim)

        # The only operation inside the cross stack is the residual CrossValue
        # recurrence.  The raw field-token stream is unchanged across layers.
        self.layers = nn.ModuleList([
            RecursiveCrossValueLayer(
                token_dim=token_dim,
                num_heads=num_heads,
                dropout=dropout,
                qk_norm=qk_norm,
            )
            for _ in range(num_layers)
        ])

        # A single post-stack mixer lets the eight cross subspaces exchange
        # information after their explicit-order recurrence is complete.
        self.post_self_attn = SDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.post_self_norm = nn.LayerNorm(token_dim)
        self.post_ffn = FeedForward(token_dim, ffn_dim, dropout)
        self.post_ffn_norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.learnable_queries)

    def forward(self, features, mask=None):
        batch_size = features.size(0)
        queries = self.learnable_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        initial_delta = self.initial_read(
            queries, features, features, mask
        )
        queries = self.initial_norm(
            queries + self.dropout(initial_delta)
        )

        for layer in self.layers:
            queries = layer(queries, features, mask)

        mixed = self.post_self_attn(queries, queries, queries)
        queries = self.post_self_norm(
            queries + self.dropout(mixed)
        )
        ffn_output = self.post_ffn(queries)
        return self.post_ffn_norm(
            queries + self.dropout(ffn_output)
        )


class RecursiveCrossValueLayer(nn.Module):
    """One exact residual step Q_(l+1) = Q_l + CrossValue(Q_l, X)."""

    def __init__(self, token_dim, num_heads, dropout=0.0, qk_norm=False):
        super().__init__()
        self.cross_attn = SDPACrossValueAttention(
            token_dim, num_heads, qk_norm
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, field_tokens, field_mask=None):
        cross_output = self.cross_attn(
            queries, field_tokens, field_tokens, field_mask
        )
        return queries + self.dropout(cross_output)


class SDPASequenceQFormerLayer(nn.Module):
    """Standard SDPA QFormer layer used only for sequence extraction."""

    def __init__(self, token_dim, num_heads, ffn_dim, dropout=0.0,
                 qk_norm=False):
        super().__init__()
        self.self_attn = SDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.self_attn_norm = nn.LayerNorm(token_dim)
        self.cross_attn = MaskedSDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.cross_attn_norm = nn.LayerNorm(token_dim)
        self.ffn = FeedForward(token_dim, ffn_dim, dropout)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, sequence_features, sequence_mask=None):
        self_output = self.self_attn(queries, queries, queries)
        queries = self.self_attn_norm(
            queries + self.dropout(self_output)
        )
        sequence_output = self.cross_attn(
            queries,
            sequence_features,
            sequence_features,
            sequence_mask,
        )
        queries = self.cross_attn_norm(
            queries + self.dropout(sequence_output)
        )
        ffn_output = self.ffn(queries)
        return self.ffn_norm(
            queries + self.dropout(ffn_output)
        )


class MaskedSDPAMultiHeadAttention(nn.Module):
    """Standard multi-head SDPA with an optional key/value padding mask."""

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

    def forward(self, query, key, value, mask=None):
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

        key_mask = None
        has_valid_key = None
        if mask is not None:
            valid = mask.to(dtype=torch.bool, device=q.device)
            key_mask = valid.view(batch_size, 1, 1, key_length)
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
