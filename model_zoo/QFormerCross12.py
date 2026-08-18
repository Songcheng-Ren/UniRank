# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

import torch
from torch import nn

from .QFormerCross8 import QFormerCross8
from .QFormerCross10 import NonRecursiveCrossValueLayer
from .QFormerCross11 import OriginalXQFormerStage


class OriginalXNSQFormerStage(nn.Module):
    """Standard NS QFormer whose every layer reads the original field X.

    The original non-sequential field stream is used to condition the initial
    learned queries and is then kept immutable as K/V for every QFormer layer.
    Only the queries are propagated between layers::

        Q_0     = LearnedQuery + Pool(X)
        Q_l'    = SelfAttention(Q_l)
        Q_l''   = CrossValue(Q_l', K=X, V=X)
        Q_(l+1) = FFN(Q_l'')
    """

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
        original_x = field_tokens
        batch_size = original_x.size(0)
        if field_mask is None:
            pooled_fields = original_x.mean(dim=1)
        else:
            valid = field_mask.to(
                dtype=original_x.dtype,
                device=original_x.device,
            ).unsqueeze(-1)
            pooled_fields = (original_x * valid).sum(dim=1)
            pooled_fields = pooled_fields / valid.sum(dim=1).clamp_min(1.0)

        context = self.context_norm(self.context_proj(pooled_fields))
        queries = self.learnable_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        queries = self.query_norm(queries + context.unsqueeze(1))
        for layer in self.layers:
            queries = layer(queries, original_x, field_mask)
        return self.final_norm(queries)


class QFormerCross12(QFormerCross8):
    """Standard QFormer layers with immutable original X in both stages.

    The NS and sequence stages both update only their queries. Every layer's
    CrossValue attention receives the corresponding stage's untouched input X
    as both K and V.
    """

    def __init__(self, feature_map, model_id="QFormerCross12", **kwargs):
        super().__init__(
            feature_map,
            model_id=model_id,
            _ns_qformer_cls=OriginalXNSQFormerStage,
            _unified_qformer_cls=OriginalXQFormerStage,
            **kwargs,
        )
