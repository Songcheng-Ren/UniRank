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


class QFormerCross(MultiTaskModel):
    """Grouped multi-emb QFormer with cross-value attention.

    Ports the production grouped-QFormer + multi-emb structure to PyTorch:

    1. Multi-emb: num_groups independent projections of user/context features,
       each producing its own user token sequence.
    2. Grouped QFormer (non-sequential): item-side features are split into
       num_groups groups. Each item group is paired with one user projection
       path, independently compressed by a QFormerStage, then all group
       outputs are concatenated and refined through full self-attention.
    3. Sequence QFormer: the item/action history is compressed by a QFormerStage
       that uses the non-sequential output as base tokens, following the same
       per-sequence → aggregation → self-attention pipeline.

    Cross-value attention is used throughout: the value is a pairwise
    query-feature interaction rather than a plain linear projection.
    """

    def __init__(self,
                 feature_map,
                 model_id="QFormerCross",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[256, 128],
                 embedding_dim=16,
                 token_dim=64,
                 num_heads=4,
                 num_layers=2,
                 num_groups=4,
                 num_queries_per_group=8,
                 seq_num_queries=4,
                 ffn_ratio=1.0,
                 num_tasks=4,
                 net_dropout=0,
                 accumulation_steps=1,
                 **kwargs):
        super(QFormerCross, self).__init__(feature_map, model_id=model_id, gpu=gpu, **kwargs)
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")

        self.feature_map = feature_map
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.accumulation_steps = accumulation_steps
        self.num_groups = num_groups

        # ---- collect feature names by source ----
        self.item_features = []
        self.user_features = []
        for feat, spec in feature_map.features.items():
            if feat in feature_map.labels or spec.get("type") == "meta":
                continue
            if spec.get("source") in ("item", "action"):
                self.item_features.append(feat)
            else:
                self.user_features.append(feat)

        self.item_info_dim = sum(
            feature_map.features[f].get("embedding_dim", embedding_dim)
            for f in self.item_features
        )
        self.non_item_dim = sum(
            feature_map.features[f].get("embedding_dim", embedding_dim)
            for f in self.user_features
        )

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # ---- per-group item feature boundaries ----
        self.item_group_bounds = self._compute_group_bounds(self.item_features, num_groups)
        self.item_group_dims = [
            self.item_group_bounds[g + 1] - self.item_group_bounds[g]
            for g in range(num_groups)
        ]

        # ---- multi-emb: num_groups independent user projections ----
        self.user_multi_proj = nn.ModuleList([
            nn.Linear(self.non_item_dim, token_dim)
            for _ in range(num_groups)
        ])
        # ---- per-group item projections (from feature subspace to token_dim) ----
        self.item_multi_proj = nn.ModuleList([
            nn.Linear(max(1, dim), token_dim)
            for dim in self.item_group_dims
        ])
        # ---- shared item projection for sequence QFormer ----
        self.item_proj = nn.Linear(self.item_info_dim, token_dim)

        # ---- grouped non-sequential QFormer ----
        self.group_qformers = nn.ModuleList([
            QFormerStage(
                token_dim=token_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                num_queries=num_queries_per_group,
                ffn_dim=int(token_dim * ffn_ratio),
                dropout=net_dropout,
            )
            for _ in range(num_groups)
        ])
        self.group_fusion_layers = nn.ModuleList([
            QFormerLayer(token_dim, num_heads, int(token_dim * ffn_ratio), net_dropout)
            for _ in range(3)
        ])
        self.group_fusion_norm = nn.LayerNorm(token_dim)

        # ---- sequence QFormer ----
        self.seq_qformer = QFormerStage(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_queries=seq_num_queries,
            ffn_dim=int(token_dim * ffn_ratio),
            dropout=net_dropout,
        )
        self.seq_fusion_layers = nn.ModuleList([
            QFormerLayer(token_dim, num_heads, int(token_dim * ffn_ratio), net_dropout)
            for _ in range(2)
        ])
        self.seq_fusion_norm = nn.LayerNorm(token_dim)

        # ---- prediction ----
        non_seq_dim = num_groups * num_queries_per_group * token_dim
        seq_dim = seq_num_queries * token_dim
        combined_dim = non_seq_dim + seq_dim
        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=combined_dim,
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
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(str(t)) for t in task]
            )
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]
            )

        self.compile(kwargs.get("dense_optimizer"), kwargs["loss"], kwargs.get("dense_learning_rate"))
        self.reset_parameters()
        self.model_to_device()

    @staticmethod
    def _compute_group_bounds(item_features, num_groups):
        """Compute balanced feature boundaries for item groups."""
        n = len(item_features)
        if n == 0:
            return [0] * (num_groups + 1)
        base = n // num_groups
        remainder = n % num_groups
        bounds = [0]
        feat_idx = 0
        for g in range(num_groups):
            size = base + (1 if g < remainder else 0)
            bounds.append(bounds[-1] + size)
        return bounds

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.size(0)

        # ---- embed item-side and user/context features ----
        item_embeddings = self.embedding_layer(item_dict, flatten_emb=True)
        item_embeddings = item_embeddings.view(batch_size, -1, self.item_info_dim)
        history_embeddings = item_embeddings[:, :-1, :]
        candidate_embedding = item_embeddings[:, -1:, :]

        context_embedding = self.embedding_layer(batch_dict, flatten_emb=True)

        # ---- multi-emb: num_groups independent user token sequences ----
        user_tokens_list = [
            self.user_multi_proj[i](context_embedding).unsqueeze(1)
            for i in range(self.num_groups)
        ]

        # ---- shared item projection for sequence QFormer ----
        history_tokens = self.item_proj(history_embeddings)

        history_mask = mask.to(dtype=history_tokens.dtype, device=history_tokens.device)

        # ---- grouped non-sequential QFormer ----
        # Each group projects its own feature subspace of item embeddings.
        group_outputs = []
        for g in range(self.num_groups):
            start, end = self.item_group_bounds[g], self.item_group_bounds[g + 1]
            if end <= start:
                continue
            item_group_history = self.item_multi_proj[g](history_embeddings[:, :, start:end])
            item_group_candidate = self.item_multi_proj[g](candidate_embedding[:, :, start:end])
            group_features = torch.cat([
                user_tokens_list[g],
                item_group_candidate,
                item_group_history,
            ], dim=1)
            group_mask = torch.cat([
                torch.ones(batch_size, 2, device=history_mask.device, dtype=history_mask.dtype),
                history_mask,
            ], dim=1)
            group_queries = self.group_qformers[g](group_features, group_mask)
            group_outputs.append(group_queries)

        x = torch.cat(group_outputs, dim=1)
        for layer in self.group_fusion_layers:
            x = layer(x, x, None)
        non_seq_queries = self.group_fusion_norm(x)

        # ---- sequence QFormer ----
        base_tokens = non_seq_queries
        seq_features = torch.cat([history_tokens, base_tokens], dim=1)
        seq_mask = torch.cat([
            history_mask,
            torch.ones(batch_size, base_tokens.size(1), device=history_mask.device, dtype=history_mask.dtype),
        ], dim=1)
        seq_queries = self.seq_qformer(seq_features, seq_mask)
        for layer in self.seq_fusion_layers:
            seq_queries = layer(seq_queries, seq_queries, None)
        seq_queries = self.seq_fusion_norm(seq_queries)

        # ---- prediction ----
        non_seq_flat = non_seq_queries.reshape(batch_size, -1)
        seq_flat = seq_queries.reshape(batch_size, -1)
        combined = torch.cat([non_seq_flat, seq_flat], dim=-1)

        tower_output = [self.tower[i](combined) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        labels = self.feature_map.labels
        return {f"{labels[i]}_pred": y_pred[i] for i in range(self.num_tasks)}


class QFormerStage(nn.Module):
    """Input-conditioned QFormer with cross-value attention.

    Queries are generated from the input features (input-conditioned), then
    iteratively refined through self-attention, cross-value attention, and
    FFN blocks.
    """

    def __init__(self, token_dim, num_heads, num_layers, num_queries, ffn_dim, dropout=0.0):
        super().__init__()
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.query_proj = nn.Linear(token_dim, num_queries * token_dim)
        self.query_norm = nn.LayerNorm(token_dim)
        self.layers = nn.ModuleList([
            QFormerLayer(token_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(self, features, mask=None):
        batch_size = features.size(0)
        pooled = features.mean(dim=1)
        queries = self.query_proj(pooled).view(batch_size, self.num_queries, self.token_dim)
        queries = self.query_norm(queries)
        for layer in self.layers:
            queries = layer(queries, features, mask)
        return self.final_norm(queries)


class QFormerLayer(nn.Module):
    def __init__(self, token_dim, num_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(token_dim)
        self.self_attn = MultiHeadAttention(token_dim, num_heads)
        self.cross_norm = nn.LayerNorm(token_dim)
        self.cross_attn = CrossValueAttention(token_dim, num_heads)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = FeedForward(token_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, features, mask=None):
        attn_out = self.self_attn(queries, queries, queries)
        queries = self.self_attn_norm(queries + self.dropout(attn_out))
        cross_out = self.cross_attn(queries, features, features, mask)
        queries = self.cross_norm(queries + self.dropout(cross_out))
        ffn_out = self.ffn(queries)
        queries = self.ffn_norm(queries + self.dropout(ffn_out))
        return queries


class MultiHeadAttention(nn.Module):
    def __init__(self, token_dim, num_heads):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.w_q = nn.Linear(token_dim, token_dim)
        self.w_k = nn.Linear(token_dim, token_dim)
        self.w_v = nn.Linear(token_dim, token_dim)
        self.w_o = nn.Linear(token_dim, token_dim)

    def forward(self, query, key, value):
        batch_size, seq_q, _ = query.shape
        seq_k = key.size(1)
        q = self.w_q(query).view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_q, -1)
        return self.w_o(out)


class CrossValueAttention(nn.Module):
    """Cross-value attention with pairwise query-feature interaction.

    Attention scores use normal QK, but the value is a pairwise interaction
    (q_i * f_j) projected and aggregated by the attention weights.
    """

    def __init__(self, token_dim, num_heads):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.w_q = nn.Linear(token_dim, token_dim)
        self.w_k = nn.Linear(token_dim, token_dim)
        self.w_qi = nn.Linear(token_dim, token_dim)
        self.w_fi = nn.Linear(token_dim, token_dim)
        self.w_v_pair = nn.Linear(token_dim, token_dim)
        self.w_o = nn.Linear(token_dim, token_dim)

    def forward(self, queries, keys, values, mask=None):
        batch_size, num_queries, _ = queries.shape
        seq_len = keys.size(1)

        q = self.w_q(queries).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(keys).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            key_mask = mask.to(dtype=scores.dtype, device=scores.device).view(batch_size, 1, 1, seq_len)
            scores = scores + (1.0 - key_mask) * -1e9
        attn = torch.softmax(scores, dim=-1)

        q_inter = self.w_qi(queries)
        f_inter = self.w_fi(keys)
        pair = q_inter.unsqueeze(2) * f_inter.unsqueeze(1)
        pair_flat = pair.reshape(-1, self.token_dim)
        v_pair = self.w_v_pair(pair_flat).view(
            batch_size, num_queries, seq_len, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)

        weighted = attn.unsqueeze(-1) * v_pair
        out = weighted.sum(dim=3)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_queries, self.token_dim)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, token_dim, ffn_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(token_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))
