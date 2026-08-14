# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

import torch
from torch import nn

from unirank.pytorch.layers import FeatureEmbedding, MLP_Block
from unirank.pytorch.models import MultiTaskModel

from .QFormerCross3 import SDPAQFormerLayer
from .QFormerCross2 import UnifiedQFormer


class QFormerCross6(MultiTaskModel):
    """Unified NS/sequence KV QFormer with pooled sequence-conditioned queries.

    V6 removes the separate non-sequential and sequence QFormer stages used by
    QFormerCross2/3.  The data flow is:

    1. Build one token for every context and target-item field (NS tokens).
    2. Build position-aware behavior tokens for the history sequence.
    3. Flatten all NS field tokens, concatenate them with the
       masked-sum-pooled sequence vector, and apply one linear projection to
       produce eight query tokens.
    4. Concatenate ``[NS tokens, sequence tokens]`` as one K/V stream and run
       SDPA QFormer layers.

    The default number of output queries is eight, matching QFormerCross3/4.
    Raw NS field tokens are still retained in the K/V stream, so the query
    bottleneck does not remove static-field information from attention.
    """

    def __init__(self,
                 feature_map,
                 model_id="QFormerCross6",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[256, 128],
                 embedding_dim=16,
                 token_dim=256,
                 num_heads=4,
                 num_layers=3,
                 ffn_ratio=4.0,
                 qk_norm=True,
                 max_len=100,
                 num_tasks=4,
                 num_queries=8,
                 net_dropout=0,
                 accumulation_steps=1,
                 **kwargs):
        super().__init__(feature_map, model_id=model_id, gpu=gpu, **kwargs)
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be positive")
        if max_len < 1:
            raise ValueError("max_len must be positive")

        self.feature_map = feature_map
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.max_len = max_len
        self.accumulation_steps = accumulation_steps

        self.target_item_features = []
        self.sequence_features = []
        self.context_features = []
        for feature, spec in feature_map.features.items():
            if feature in feature_map.labels or spec.get("type") == "meta":
                continue
            if spec.get("source") == "item":
                self.target_item_features.append(feature)
                self.sequence_features.append(feature)
            elif spec.get("source") == "action":
                # Actions are history-only features.  The final item position
                # is a zero placeholder supplied by the dataloader.
                self.sequence_features.append(feature)
            else:
                self.context_features.append(feature)

        if not self.target_item_features:
            raise ValueError("QFormerCross6 requires target item features")
        if not self.sequence_features:
            raise ValueError("QFormerCross6 requires item/action sequence features")
        if not self.context_features:
            raise ValueError("QFormerCross6 requires user/context features")

        self.ns_features = self.context_features + self.target_item_features
        self.num_ns_tokens = len(self.ns_features)
        if num_queries < 1:
            raise ValueError("num_queries must be positive")
        self.num_queries = num_queries

        self.item_info_dim = sum(
            feature_map.features[feature].get("embedding_dim", embedding_dim)
            for feature in self.sequence_features
        )
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # Preserve field identity before the query projection.
        self.field_projection = nn.ModuleDict({
            feature: nn.Linear(
                feature_map.features[feature].get("embedding_dim", embedding_dim),
                token_dim,
                bias=False,
            )
            for feature in self.ns_features
        })
        self.ns_field_embedding = nn.Parameter(
            torch.empty(self.num_ns_tokens, token_dim)
        )
        self.ns_input_norm = nn.LayerNorm(token_dim)

        self.sequence_projection = nn.Linear(self.item_info_dim, token_dim)
        # Positional parameters remain dense parameters rather than sparse
        # embedding tables, matching QFormerCross2/3.
        self.position_embedding = nn.Parameter(torch.empty(max_len, token_dim))
        self.sequence_input_norm = nn.LayerNorm(token_dim)

        # Direct V6 query construction: flatten all raw NS tokens, append the
        # sequence sum-pool, then use exactly one Linear to emit 8 queries.
        self.query_projection = nn.Linear(
            (self.num_ns_tokens + 1) * token_dim,
            self.num_queries * token_dim,
        )
        self.unified_qformer = UnifiedQFormer(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=int(token_dim * ffn_ratio),
            dropout=net_dropout,
            qk_norm=qk_norm,
            layer_cls=SDPAQFormerLayer,
        )

        tower_input_dim = self.num_queries * token_dim
        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=tower_input_dim,
                output_dim=1,
                hidden_units=tower_hidden_units,
                hidden_activations=tower_activations,
                output_activation=None,
                dropout_rates=net_dropout,
            )
            for _ in range(num_tasks)
        ])
        if isinstance(task, list):
            if len(task) != num_tasks:
                raise ValueError("the number of tasks must equal the length of task")
            self.output_activation = nn.ModuleList([
                self.get_output_activation(str(task_type))
                for task_type in task
            ])
        else:
            self.output_activation = nn.ModuleList([
                self.get_output_activation(task)
                for _ in range(num_tasks)
            ])

        self.compile(
            kwargs.get("dense_optimizer"),
            kwargs["loss"],
            kwargs.get("dense_learning_rate"),
        )
        self.reset_parameters()
        nn.init.normal_(self.ns_field_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        self.model_to_device()

    def _build_non_sequence_tokens(self, context_embedding_dict,
                                   item_embedding_dict):
        tokens = []
        for feature in self.context_features:
            embedding = context_embedding_dict[feature]
            if embedding.dim() != 2:
                raise ValueError(
                    f"context feature {feature} must produce B x D embeddings, "
                    f"got shape={tuple(embedding.shape)}"
                )
            tokens.append(self.field_projection[feature](embedding))

        for feature in self.target_item_features:
            embedding = item_embedding_dict[feature]
            if embedding.dim() != 3:
                raise ValueError(
                    f"item feature {feature} must produce B x (S+1) x D embeddings, "
                    f"got shape={tuple(embedding.shape)}"
                )
            tokens.append(self.field_projection[feature](embedding[:, -1, :]))

        ns_tokens = torch.stack(tokens, dim=1)
        ns_tokens = ns_tokens + self.ns_field_embedding.unsqueeze(0)
        return self.ns_input_norm(ns_tokens)

    def _build_sequence_tokens(self, item_embedding_dict, history_mask):
        history_fields = [
            item_embedding_dict[feature][:, :-1, :]
            for feature in self.sequence_features
        ]
        history_embedding = torch.cat(history_fields, dim=-1)
        sequence_length = history_embedding.size(1)
        if sequence_length > self.max_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_len={self.max_len}"
            )

        sequence_tokens = self.sequence_projection(history_embedding)
        valid_mask = history_mask.to(dtype=torch.bool, device=sequence_tokens.device)
        position_ids = valid_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        position_tokens = self.position_embedding[position_ids]
        sequence_tokens = self.sequence_input_norm(
            sequence_tokens + position_tokens
        )
        return sequence_tokens * valid_mask.unsqueeze(-1).to(
            dtype=sequence_tokens.dtype
        )

    def _build_queries(self, ns_tokens, sequence_tokens, history_mask):
        """Generate Q from flattened NS tokens + masked sum(sequence)."""
        valid = history_mask.to(
            dtype=sequence_tokens.dtype, device=sequence_tokens.device
        ).unsqueeze(-1)
        sequence_sum = (sequence_tokens * valid).sum(dim=1)
        query_input = torch.cat([
            ns_tokens.reshape(ns_tokens.size(0), -1),
            sequence_sum,
        ], dim=-1)
        queries = self.query_projection(query_input)
        return queries.view(ns_tokens.size(0), self.num_queries, self.token_dim)

    def _build_kv_tokens(self, ns_tokens, sequence_tokens, history_mask):
        """Concatenate NS and sequence tokens into the shared K/V stream."""
        batch_size = ns_tokens.size(0)
        ns_mask = torch.ones(
            batch_size,
            ns_tokens.size(1),
            dtype=history_mask.dtype,
            device=history_mask.device,
        )
        kv_tokens = torch.cat([ns_tokens, sequence_tokens], dim=1)
        kv_mask = torch.cat([ns_mask, history_mask], dim=1)
        return kv_tokens, kv_mask

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.size(0)
        if mask.size(1) > self.max_len:
            raise ValueError(
                f"input history length {mask.size(1)} exceeds max_len={self.max_len}"
            )

        context_embedding_dict = self.embedding_layer.embedding_layer(batch_dict)
        item_embedding_dict = self.embedding_layer.embedding_layer(item_dict)

        ns_tokens = self._build_non_sequence_tokens(
            context_embedding_dict, item_embedding_dict
        )
        history_mask = mask.to(device=ns_tokens.device)
        sequence_tokens = self._build_sequence_tokens(
            item_embedding_dict, history_mask
        )
        queries = self._build_queries(ns_tokens, sequence_tokens, history_mask)
        kv_tokens, kv_mask = self._build_kv_tokens(
            ns_tokens, sequence_tokens, history_mask
        )

        queries = self.activation_checkpoint(
            self.unified_qformer,
            queries,
            kv_tokens,
            kv_mask,
        )

        bottom_output = queries.reshape(batch_size, -1)
        tower_output = [
            self.tower[index](bottom_output)
            for index in range(self.num_tasks)
        ]
        y_pred = [
            self.output_activation[index](tower_output[index])
            for index in range(self.num_tasks)
        ]
        labels = self.feature_map.labels
        return {
            f"{labels[index]}_pred": y_pred[index]
            for index in range(self.num_tasks)
        }
