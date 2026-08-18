# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

import math

import torch
from torch import nn
import torch.nn.functional as F

from unirank.pytorch.layers import FeatureEmbedding, MLP_Block
from unirank.pytorch.models import MultiTaskModel


class FeedForward(nn.Module):
    """Two-layer feed-forward block shared by both QFormer stages."""

    def __init__(self, token_dim, ffn_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(token_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, token_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, inputs):
        return self.dropout(self.fc2(F.gelu(self.fc1(inputs))))


class SDPAMultiHeadAttention(nn.Module):
    """Unmasked multi-head attention backed by PyTorch SDPA."""

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
        self.q_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )

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


class MaskedSDPAMultiHeadAttention(nn.Module):
    """Multi-head SDPA with an optional key/value padding mask."""

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
        self.q_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )

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


class RecursiveCrossValueLayer(nn.Module):
    """One exact residual step: Q_(l+1) = Q_l + CrossValue(Q_l, X)."""

    def __init__(self, token_dim, num_heads, dropout=0.0, qk_norm=False):
        super().__init__()
        self.cross_attn = SDPACrossValueAttention(
            token_dim, num_heads, qk_norm
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, features, mask=None):
        cross_output = self.cross_attn(queries, features, features, mask)
        return queries + self.dropout(cross_output)


class RecursiveCrossValueStage(nn.Module):
    """Read fields once, recursively cross them, then mix the queries."""

    def __init__(self, token_dim, num_heads, num_layers, num_queries, ffn_dim,
                 dropout=0.0, qk_norm=False):
        super().__init__()
        self.token_dim = token_dim
        self.num_queries = num_queries
        self.learnable_queries = nn.Parameter(
            torch.empty(num_queries, token_dim)
        )
        self.initial_read = MaskedSDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.initial_norm = nn.LayerNorm(token_dim)
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
        queries = self.post_self_norm(queries + self.dropout(mixed))
        ffn_output = self.post_ffn(queries)
        return self.post_ffn_norm(
            queries + self.dropout(ffn_output)
        )


class RecursiveSequenceCrossValueStage(nn.Module):
    """Apply the same residual CrossValue recurrence to sequence tokens."""

    def __init__(self, token_dim, num_heads, num_layers, ffn_dim,
                 dropout=0.0, qk_norm=False):
        super().__init__()
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
        queries = self.post_self_norm(queries + self.dropout(mixed))
        ffn_output = self.post_ffn(queries)
        return self.post_ffn_norm(
            queries + self.dropout(ffn_output)
        )


class QFormerCross8(MultiTaskModel):
    """Symmetric CrossValue recurrences for fields and behavior sequences.

    The first stage initializes latent queries from user/context and target
    item fields, then recursively crosses the queries with the unchanged field
    tokens. The second stage applies the same recurrence to the unchanged
    sequence-token stream::

        Q_ns_(l+1)  = Q_ns_l  + CrossValue(Q_ns_l, X)
        Q_seq_(l+1) = Q_seq_l + CrossValue(Q_seq_l, S)

    Each stage mixes queries and applies an FFN only after its complete cross
    stack. This file contains the complete model implementation and does not
    depend on another QFormerCross variant.
    """

    def __init__(self,
                 feature_map,
                 model_id="QFormerCross8",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[256, 128],
                 embedding_dim=16,
                 token_dim=256,
                 num_heads=4,
                 num_queries=8,
                 num_ns_layers=2,
                 num_unified_layers=3,
                 ffn_ratio=4.0,
                 qk_norm=True,
                 max_len=100,
                 num_tasks=4,
                 net_dropout=0,
                 accumulation_steps=1,
                 _ns_qformer_cls=None,
                 _unified_qformer_cls=None,
                 **kwargs):
        super().__init__(feature_map, model_id=model_id, gpu=gpu, **kwargs)
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if num_queries < 1:
            raise ValueError("num_queries must be positive")
        if num_ns_layers < 1 or num_unified_layers < 1:
            raise ValueError("both QFormer stages require at least one layer")
        if ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be positive")
        if max_len < 1:
            raise ValueError("max_len must be positive")

        self.feature_map = feature_map
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_queries = num_queries
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
                # Target actions are zero placeholders and only belong to the
                # history stream.
                self.sequence_features.append(feature)
            else:
                self.context_features.append(feature)
        if not self.target_item_features:
            raise ValueError("QFormerCross8 requires target item features")
        if not self.sequence_features:
            raise ValueError(
                "QFormerCross8 requires item/action sequence features"
            )
        if not self.context_features:
            raise ValueError("QFormerCross8 requires user/context features")

        self.ns_features = self.context_features + self.target_item_features
        self.item_info_dim = sum(
            feature_map.features[feature].get(
                "embedding_dim", embedding_dim
            )
            for feature in self.sequence_features
        )

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.field_projection = nn.ModuleDict({
            feature: nn.Linear(
                feature_map.features[feature].get(
                    "embedding_dim", embedding_dim
                ),
                token_dim,
                bias=False,
            )
            for feature in self.ns_features
        })
        self.ns_field_embedding = nn.Parameter(torch.empty(
            len(self.ns_features), token_dim
        ))
        self.ns_input_norm = nn.LayerNorm(token_dim)

        self.sequence_projection = nn.Linear(self.item_info_dim, token_dim)
        # A Parameter keeps this small position table in the dense optimizer.
        self.position_embedding = nn.Parameter(torch.empty(max_len, token_dim))
        self.sequence_input_norm = nn.LayerNorm(token_dim)

        ffn_dim = int(token_dim * ffn_ratio)
        ns_qformer_cls = _ns_qformer_cls or RecursiveCrossValueStage
        self.ns_qformer = ns_qformer_cls(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_ns_layers,
            num_queries=num_queries,
            ffn_dim=ffn_dim,
            dropout=net_dropout,
            qk_norm=qk_norm,
        )
        unified_qformer_cls = (
            _unified_qformer_cls or RecursiveSequenceCrossValueStage
        )
        self.unified_qformer = unified_qformer_cls(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_unified_layers,
            ffn_dim=ffn_dim,
            dropout=net_dropout,
            qk_norm=qk_norm,
        )

        tower_input_dim = num_queries * token_dim
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
                raise ValueError(
                    "the number of tasks must equal the length of task"
                )
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
                    f"item feature {feature} must produce B x (S+1) x D "
                    f"embeddings, got shape={tuple(embedding.shape)}"
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
                f"sequence length {sequence_length} exceeds "
                f"max_len={self.max_len}"
            )

        sequence_tokens = self.sequence_projection(history_embedding)
        # Count valid chronological events so left padding does not alter the
        # position assigned to real behavior tokens.
        valid_mask = history_mask.to(dtype=torch.bool)
        position_ids = valid_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        position_tokens = self.position_embedding[position_ids]
        sequence_tokens = self.sequence_input_norm(
            sequence_tokens + position_tokens
        )
        return sequence_tokens * valid_mask.unsqueeze(-1).to(
            dtype=sequence_tokens.dtype
        )

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.size(0)
        if mask.size(1) > self.max_len:
            raise ValueError(
                f"input history length {mask.size(1)} exceeds "
                f"max_len={self.max_len}"
            )

        context_embedding_dict = self.embedding_layer.embedding_layer(
            batch_dict
        )
        item_embedding_dict = self.embedding_layer.embedding_layer(item_dict)

        ns_field_tokens = self._build_non_sequence_tokens(
            context_embedding_dict, item_embedding_dict
        )
        ns_mask = torch.ones(
            batch_size,
            ns_field_tokens.size(1),
            dtype=mask.dtype,
            device=mask.device,
        )
        ns_queries = self.activation_checkpoint(
            self.ns_qformer, ns_field_tokens, ns_mask
        )

        history_mask = mask.to(device=ns_queries.device)
        sequence_tokens = self._build_sequence_tokens(
            item_embedding_dict, history_mask
        )
        unified_queries = self.activation_checkpoint(
            self.unified_qformer,
            ns_queries,
            sequence_tokens,
            history_mask,
        )

        bottom_output = unified_queries.reshape(batch_size, -1)
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
