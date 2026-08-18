# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

from torch import nn

from .QFormerCross8 import QFormerCross8
from .QFormerCross10 import (
    NonRecursiveCrossValueLayer,
    NonRecursiveCrossValueStage,
)


class OriginalXQFormerStage(nn.Module):
    """Standard QFormer layers that always cross against the original X.

    Queries evolve from layer to layer, but ``source_tokens`` is never
    overwritten. Consequently every CrossValue attention receives the same
    original stage input as both its key and value source::

        Q_l'    = SelfAttention(Q_l)
        Q_l''   = CrossValue(Q_l', K=X, V=X)
        Q_(l+1) = FFN(Q_l'')
    """

    def __init__(self, token_dim, num_heads, num_layers, ffn_dim,
                 dropout=0.0, qk_norm=False):
        super().__init__()
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

    def forward(self, queries, source_tokens, source_mask=None):
        # Keep this reference fixed: only queries are updated between layers.
        original_x = source_tokens
        for layer in self.layers:
            queries = layer(queries, original_x, source_mask)
        return self.final_norm(queries)


class QFormerCross11(QFormerCross8):
    """Original QFormer order with immutable CrossValue source streams.

    Both stages restore the per-layer Self-Attention -> Cross-Attention -> FFN
    structure. Every cross-attention layer uses the unmodified stage input X
    as V (and K); only the latent queries are propagated across layers.
    """

    def __init__(self, feature_map, model_id="QFormerCross11", **kwargs):
        super().__init__(
            feature_map,
            model_id=model_id,
            _ns_qformer_cls=NonRecursiveCrossValueStage,
            _unified_qformer_cls=OriginalXQFormerStage,
            **kwargs,
        )
