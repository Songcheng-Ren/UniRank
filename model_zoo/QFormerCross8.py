# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

from torch import nn

from .QFormerCross import FeedForward
from .QFormerCross2 import QFormerCross2
from .QFormerCross3 import SDPAMultiHeadAttention
from .QFormerCross7 import (
    RecursiveCrossValueLayer,
    RecursiveCrossValueStage,
)


class QFormerCross8(QFormerCross2):
    """Symmetric CrossValue recurrences for fields and sequence.

    The non-sequential encoder is identical to QFormerCross7.  The crossed
    field queries then become the initial state of a second recurrence over
    the unchanged sequence-token stream::

        Q_ns_(l+1)  = Q_ns_l  + CrossValue(Q_ns_l, X)
        Q_seq_(l+1) = Q_seq_l + CrossValue(Q_seq_l, S)

    Each stage applies query mixing and an FFN only after its complete cross
    stack.  This keeps one bilinear operator as the structural primitive for
    both explicit field crossing and query-conditioned sequence crossing.
    """

    def __init__(self, feature_map, model_id="QFormerCross8", **kwargs):
        super().__init__(
            feature_map,
            model_id=model_id,
            _qformer_stage_cls=RecursiveCrossValueStage,
            _unified_qformer_cls=RecursiveSequenceCrossValueStage,
            **kwargs,
        )


class RecursiveSequenceCrossValueStage(nn.Module):
    """Apply the exact residual CrossValue recurrence over sequence tokens."""

    def __init__(self, token_dim, num_heads, num_layers, ffn_dim,
                 dropout=0.0, qk_norm=False, layer_cls=None):
        super().__init__()
        # layer_cls is accepted for compatibility with QFormerCross2's stage
        # construction hook; V8 deliberately fixes the recurrence primitive.
        del layer_cls
        self.layers = nn.ModuleList([
            RecursiveCrossValueLayer(
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

    def forward(self, queries, sequence_features, sequence_mask=None):
        for layer in self.layers:
            queries = layer(queries, sequence_features, sequence_mask)

        mixed = self.post_self_attn(queries, queries, queries)
        queries = self.post_self_norm(
            queries + self.dropout(mixed)
        )
        ffn_output = self.post_ffn(queries)
        return self.post_ffn_norm(
            queries + self.dropout(ffn_output)
        )

