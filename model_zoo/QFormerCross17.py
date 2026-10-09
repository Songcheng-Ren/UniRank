# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

"""Cross8's two QFormer stages packaged into stackable fixed-output blocks."""

from functools import partial

from torch import nn

from .QFormerCross8 import (
    QFormerCross8,
    RecursiveCrossValueStage,
    RecursiveSequenceCrossValueStage,
)


class UnifiedQFormerBlock(nn.Module):
    """NS fields -> latent tokens -> sequence read -> fixed-size vector.

    Both stages use Cross8's exact attention, residual, normalization and FFN
    implementations. Stage 2 carries forward Stage 1's queries; it does not
    initialize another set of learned queries. A single linear projection
    maps the flattened sequence-conditioned queries from M*D to output_dim.

    Each block owns its parameters. Its output is consumed by the prediction
    towers or projected to input tokens for the next independently parameterized
    block in an ordinary feed-forward stack.
    """

    def __init__(self, token_dim, num_heads, num_queries, num_ns_layers,
                 num_unified_layers, ffn_dim, output_dim, dropout=0.0,
                 qk_norm=True):
        super().__init__()
        if not isinstance(output_dim, int) or output_dim < 1:
            raise ValueError("output_dim must be a positive integer")
        self.num_queries = num_queries
        self.token_dim = token_dim
        self.output_dim = output_dim
        self.ns_qformer = RecursiveCrossValueStage(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_ns_layers,
            num_queries=num_queries,
            ffn_dim=ffn_dim,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.sequence_qformer = RecursiveSequenceCrossValueStage(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_unified_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.output_projection = nn.Linear(
            num_queries * token_dim, output_dim
        )

    def forward(self, ns_tokens, sequence_tokens, ns_mask=None,
                sequence_mask=None):
        ns_queries = self.ns_qformer(ns_tokens, ns_mask)
        sequence_queries = self.sequence_qformer(
            ns_queries, sequence_tokens, sequence_mask
        )
        return self.output_projection(sequence_queries.flatten(start_dim=1))


class StackedUnifiedQFormerBlocks(nn.Module):
    """Stack independent unified blocks with vector-to-token transitions.

    The first block reads the original non-sequential field tokens. Between
    blocks, an independent Linear(output_dim, M*D) restores M latent input
    tokens. Every block reads the same original sequence as K/V, with its own
    projections and query parameters. Only the final vector feeds the towers.
    """

    def __init__(self, token_dim, num_heads, num_queries, num_ns_layers,
                 num_unified_layers, ffn_dim, output_dim, num_blocks,
                 dropout=0.0, qk_norm=True):
        super().__init__()
        if not isinstance(num_blocks, int) or num_blocks < 1:
            raise ValueError("num_blocks must be a positive integer")
        self.num_blocks = num_blocks
        self.num_queries = num_queries
        self.token_dim = token_dim
        self.output_dim = output_dim
        self.blocks = nn.ModuleList([
            UnifiedQFormerBlock(
                token_dim=token_dim,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns_layers=num_ns_layers,
                num_unified_layers=num_unified_layers,
                ffn_dim=ffn_dim,
                output_dim=output_dim,
                dropout=dropout,
                qk_norm=qk_norm,
            )
            for _ in range(num_blocks)
        ])
        self.token_projections = nn.ModuleList([
            nn.Linear(output_dim, num_queries * token_dim)
            for _ in range(num_blocks - 1)
        ])

    def forward(self, ns_tokens, sequence_tokens, ns_mask=None,
                sequence_mask=None):
        output = self.blocks[0](
            ns_tokens, sequence_tokens, ns_mask, sequence_mask
        )
        for projection, block in zip(self.token_projections, self.blocks[1:]):
            next_tokens = projection(output).reshape(
                output.size(0), self.num_queries, self.token_dim
            )
            output = block(
                next_tokens,
                sequence_tokens,
                ns_mask=None,
                sequence_mask=sequence_mask,
            )
        return output


class QFormerCross17(QFormerCross8):
    """Cross8 with independent unified blocks and fixed-dimensional output.

    With the default settings, F*256 non-sequence field tokens are compressed
    into 8*256 queries, enriched by the sequence stage, and linearly projected
    to a 256-dimensional vector. With num_blocks > 1, successive vectors are
    projected into 8*256 tokens for the next block. All task towers consume the
    final block's vector. Blocks execute once each and do not share parameters.
    """

    def __init__(self, feature_map, model_id="QFormerCross17", output_dim=256,
                 num_blocks=1, **kwargs):
        if not isinstance(num_blocks, int) or num_blocks < 1:
            raise ValueError("num_blocks must be a positive integer")
        block_cls = (
            UnifiedQFormerBlock if num_blocks == 1 else partial(
                StackedUnifiedQFormerBlocks, num_blocks=num_blocks
            )
        )
        super().__init__(
            feature_map,
            model_id=model_id,
            output_dim=output_dim,
            _interaction_block_cls=block_cls,
            **kwargs,
        )
        self.num_blocks = num_blocks
