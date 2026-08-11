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
    4. Target-aware shortcut: user/context, target, attentive history and their
       target-history interaction bypass the query bottleneck and are supplied
       directly to both the sequence QFormer and task towers.

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
                 qk_norm=True,
                 use_target_aware_shortcut=True,
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
        self.use_target_aware_shortcut = use_target_aware_shortcut

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
        if not self.item_features:
            raise ValueError("QFormerCross requires at least one item/action feature")
        if not self.user_features:
            raise ValueError("QFormerCross requires at least one user/context feature")
        if not 1 <= num_groups <= len(self.item_features):
            raise ValueError(
                "num_groups must be in [1, number of item/action features], "
                f"got num_groups={num_groups}, features={len(self.item_features)}"
            )

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # ---- per-group item feature boundaries ----
        item_feature_dims = [
            feature_map.features[f].get("embedding_dim", embedding_dim)
            for f in self.item_features
        ]
        self.item_group_bounds = self._compute_group_bounds(item_feature_dims, num_groups)
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
            nn.Linear(dim, token_dim)
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
                qk_norm=qk_norm,
            )
            for _ in range(num_groups)
        ])
        self.group_fusion_layers = nn.ModuleList([
            QFormerLayer(
                token_dim, num_heads, int(token_dim * ffn_ratio), net_dropout, qk_norm
            )
            for _ in range(1)
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
            qk_norm=qk_norm,
        )
        self.seq_fusion_layers = nn.ModuleList([
            QFormerLayer(
                token_dim, num_heads, int(token_dim * ffn_ratio), net_dropout, qk_norm
            )
            for _ in range(1)
        ])
        self.seq_fusion_norm = nn.LayerNorm(token_dim)

        # ---- target-aware residual path ----
        # The QFormer queries are deliberately narrow information bottlenecks.
        # Preserve a direct route for user, target and target-conditioned history
        # features so the prediction tower does not need to reconstruct them from
        # compressed queries alone.
        if use_target_aware_shortcut:
            self.context_shortcut_proj = nn.Linear(self.non_item_dim, token_dim)
            self.target_query_norm = nn.LayerNorm(token_dim)
            self.history_key_norm = nn.LayerNorm(token_dim)
            self.shortcut_fusion = nn.Sequential(
                nn.Linear(4 * token_dim, token_dim),
                nn.GELU(),
                nn.LayerNorm(token_dim),
            )

        # ---- prediction ----
        non_seq_dim = num_groups * num_queries_per_group * token_dim
        seq_dim = seq_num_queries * token_dim
        shortcut_dim = token_dim if use_target_aware_shortcut else 0
        combined_dim = non_seq_dim + seq_dim + shortcut_dim
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
    def _compute_group_bounds(feature_dims, num_groups):
        """Return flattened embedding offsets for balanced field groups.

        ``FeatureEmbedding(..., flatten_emb=True)`` concatenates embedding
        dimensions, not field indices. The old implementation returned field
        indices and therefore sliced only a few scalar dimensions from the
        flattened tensor (for example ``[0:2]`` instead of ``[0:32]`` for two
        16-dimensional fields).
        """
        n = len(feature_dims)
        if n == 0:
            raise ValueError("feature_dims must not be empty")
        if not 1 <= num_groups <= n:
            raise ValueError("num_groups must be between 1 and len(feature_dims)")
        dim_offsets = [0]
        for dim in feature_dims:
            if dim <= 0:
                raise ValueError(f"feature embedding dimensions must be positive, got {dim}")
            dim_offsets.append(dim_offsets[-1] + dim)

        base = n // num_groups
        remainder = n % num_groups
        bounds = [0]
        feat_idx = 0
        for g in range(num_groups):
            size = base + (1 if g < remainder else 0)
            feat_idx += size
            bounds.append(dim_offsets[feat_idx])
        return bounds

    def _target_aware_shortcut(self, context_embedding, candidate_token,
                               history_tokens, history_mask):
        target_query = self.target_query_norm(candidate_token)
        history_keys = self.history_key_norm(history_tokens)
        valid = history_mask.to(
            dtype=torch.bool, device=history_tokens.device
        ).view(history_tokens.size(0), 1, 1, history_tokens.size(1))
        history_interest = F.scaled_dot_product_attention(
            target_query.unsqueeze(1),
            history_keys.unsqueeze(1),
            history_tokens.unsqueeze(1),
            attn_mask=valid,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(self.token_dim),
        ).squeeze(1)

        context_token = self.context_shortcut_proj(context_embedding).unsqueeze(1)
        shortcut_input = torch.cat([
            context_token,
            candidate_token,
            history_interest,
            candidate_token * history_interest,
        ], dim=-1)
        return self.shortcut_fusion(shortcut_input)

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
        candidate_token = self.item_proj(candidate_embedding)

        history_mask = mask.to(dtype=history_tokens.dtype, device=history_tokens.device)

        shortcut_token = None
        if self.use_target_aware_shortcut:
            shortcut_token = self._target_aware_shortcut(
                context_embedding, candidate_token, history_tokens, history_mask
            )

        # ---- grouped non-sequential QFormer ----
        # Each group projects its own feature subspace of item embeddings.
        group_outputs = []
        for g in range(self.num_groups):
            start, end = self.item_group_bounds[g], self.item_group_bounds[g + 1]
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
            group_queries = self.activation_checkpoint(
                self.group_qformers[g], group_features, group_mask
            )
            group_outputs.append(group_queries)

        x = torch.cat(group_outputs, dim=1)
        for layer in self.group_fusion_layers:
            x = layer(x, x, None)
        non_seq_queries = self.group_fusion_norm(x)

        # ---- sequence QFormer ----
        base_tokens = non_seq_queries
        seq_feature_parts = [history_tokens, base_tokens]
        if shortcut_token is not None:
            seq_feature_parts.append(shortcut_token)
        seq_features = torch.cat(seq_feature_parts, dim=1)
        seq_mask = torch.cat([
            history_mask,
            torch.ones(
                batch_size,
                seq_features.size(1) - history_tokens.size(1),
                device=history_mask.device,
                dtype=history_mask.dtype,
            ),
        ], dim=1)
        seq_queries = self.activation_checkpoint(
            self.seq_qformer, seq_features, seq_mask
        )
        for layer in self.seq_fusion_layers:
            seq_queries = layer(seq_queries, seq_queries, None)
        seq_queries = self.seq_fusion_norm(seq_queries)

        # ---- prediction ----
        non_seq_flat = non_seq_queries.reshape(batch_size, -1)
        seq_flat = seq_queries.reshape(batch_size, -1)
        prediction_parts = [non_seq_flat, seq_flat]
        if shortcut_token is not None:
            prediction_parts.append(shortcut_token.squeeze(1))
        combined = torch.cat(prediction_parts, dim=-1)

        tower_output = [self.tower[i](combined) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        labels = self.feature_map.labels
        return {f"{labels[i]}_pred": y_pred[i] for i in range(self.num_tasks)}


class QFormerStage(nn.Module):
    """QFormer with learnable queries + input conditioning.

    Queries are initialized as learnable parameters, then modulated by a
    context vector derived from the input features (input-conditioned).
    This preserves spatial information better than mean-pool → Linear.
    """

    def __init__(self, token_dim, num_heads, num_layers, num_queries, ffn_dim,
                 dropout=0.0, qk_norm=False):
        super().__init__()
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.learnable_queries = nn.Parameter(torch.empty(num_queries, token_dim))
        self.context_proj = nn.Linear(token_dim, token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.query_norm = nn.LayerNorm(token_dim)
        self.layers = nn.ModuleList([
            QFormerLayer(token_dim, num_heads, ffn_dim, dropout, qk_norm)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(token_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.learnable_queries)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

    def forward(self, features, mask=None):
        batch_size = features.size(0)
        if mask is None:
            pooled_features = features.mean(dim=1)
        else:
            valid = mask.to(dtype=features.dtype, device=features.device).unsqueeze(-1)
            pooled_features = (features * valid).sum(dim=1)
            pooled_features = pooled_features / valid.sum(dim=1).clamp_min(1.0)
        context = self.context_proj(pooled_features)
        context = self.context_norm(context)
        queries = self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + context.unsqueeze(1)
        queries = self.query_norm(queries)
        for layer in self.layers:
            queries = layer(queries, features, mask)
        return self.final_norm(queries)


class QFormerLayer(nn.Module):
    def __init__(self, token_dim, num_heads, ffn_dim, dropout=0.0, qk_norm=False):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(token_dim)
        self.self_attn = MultiHeadAttention(token_dim, num_heads, qk_norm)
        self.cross_norm = nn.LayerNorm(token_dim)
        self.cross_attn = CrossValueAttention(token_dim, num_heads, qk_norm)
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
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
        for m in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, query, key, value):
        batch_size, seq_q, _ = query.shape
        seq_k = key.size(1)
        q = self.w_q(query).view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_q, -1)
        return self.w_o(out)


class CrossValueAttention(nn.Module):
    """SDPA-backed explicit query-feature interaction.

    Q, K and feature values use standard head-local layouts so PyTorch can
    dispatch scaled_dot_product_attention to a fused kernel. After SDPA pools
    the relevant feature value for each query and head, an explicit
    query-feature product is formed and projected back to token_dim.
    """

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
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
        for m in [self.w_q, self.w_k, self.w_qi, self.w_fi, self.w_v_pair, self.w_o]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, queries, keys, values, mask=None):
        batch_size, num_queries, _ = queries.shape
        seq_len = keys.size(1)

        q = self.w_q(queries).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(keys).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_fi(values).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        key_mask = None
        if mask is not None:
            key_mask = mask.to(dtype=torch.bool, device=q.device).view(
                batch_size, 1, 1, seq_len
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

        q_inter = self.w_qi(queries).view(
            batch_size, num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        pair_summary = q_inter * attended_features
        pair_summary = pair_summary.transpose(1, 2).contiguous().view(
            batch_size, num_queries, self.token_dim
        )
        out = self.w_v_pair(pair_summary)
        return self.w_o(out)


class FeedForward(nn.Module):
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

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))
