# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

import torch
from torch import nn

from .QFormerCross8 import (
    FeedForward,
    QFormerCross8,
    SDPACrossValueAttention,
    SDPAMultiHeadAttention,
)


class NonRecursiveCrossValueLayer(nn.Module):
    """A regular QFormer layer without the explicit CrossValue recurrence.

    Query mixing, field crossing and the FFN are all applied inside every
    layer. This differs from V8's order-growing stack, whose inner layers only
    perform ``Q = Q + CrossValue(Q, X)`` and postpone mixing until the end.
    """

    def __init__(self, token_dim, num_heads, ffn_dim, dropout=0.0,
                 qk_norm=False):
        super().__init__()
        self.self_attn = SDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.self_attn_norm = nn.LayerNorm(token_dim)
        self.cross_attn = SDPACrossValueAttention(
            token_dim, num_heads, qk_norm
        )
        self.cross_attn_norm = nn.LayerNorm(token_dim)
        self.ffn = FeedForward(token_dim, ffn_dim, dropout)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, field_tokens, field_mask=None):
        self_output = self.self_attn(queries, queries, queries)
        queries = self.self_attn_norm(
            queries + self.dropout(self_output)
        )
        cross_output = self.cross_attn(
            queries, field_tokens, field_tokens, field_mask
        )
        queries = self.cross_attn_norm(
            queries + self.dropout(cross_output)
        )
        ffn_output = self.ffn(queries)
        return self.ffn_norm(
            queries + self.dropout(ffn_output)
        )


class NonRecursiveCrossValueStage(nn.Module):
    """Input-conditioned queries followed by regular CrossValue layers."""

    def __init__(self, token_dim, num_heads, num_layers, num_queries, ffn_dim,
                 dropout=0.0, qk_norm=False):
        super().__init__()
        self.num_queries = num_queries
        self.learnable_queries = nn.Parameter(
            torch.empty(num_queries, token_dim)
        )
        self.context_proj = nn.Linear(token_dim, token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.query_norm = nn.LayerNorm(token_dim)
        self.layers = nn.ModuleList([
            NonRecursiveCrossValueLayer(
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

    def forward(self, field_tokens, field_mask=None):
        batch_size = field_tokens.size(0)
        if field_mask is None:
            pooled_fields = field_tokens.mean(dim=1)
        else:
            valid = field_mask.to(
                dtype=field_tokens.dtype,
                device=field_tokens.device,
            ).unsqueeze(-1)
            pooled_fields = (field_tokens * valid).sum(dim=1)
            pooled_fields = pooled_fields / valid.sum(dim=1).clamp_min(1.0)

        context = self.context_norm(self.context_proj(pooled_fields))
        queries = self.learnable_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        queries = self.query_norm(queries + context.unsqueeze(1))
        for layer in self.layers:
            queries = layer(queries, field_tokens, field_mask)
        return self.final_norm(queries)


class QFormerCross10(QFormerCross8):
    """Non-recursive field crossing with recursive sequence crossing.

    The non-sequential stage uses regular QFormer blocks, with self-attention,
    CrossValue and FFN processing in every layer. The sequential stage keeps
    V8's exact residual CrossValue recurrence unchanged.
    """

    def __init__(self, feature_map, model_id="QFormerCross10", **kwargs):
        super().__init__(
            feature_map,
            model_id=model_id,
            _ns_qformer_cls=NonRecursiveCrossValueStage,
            **kwargs,
        )
