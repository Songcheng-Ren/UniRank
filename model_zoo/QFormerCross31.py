# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# =========================================================================

import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from .QFormerCross import FeedForward
from .QFormerCross2 import QFormerCross2
from .QFormerCross3 import SDPAMultiHeadAttention, SDPAQFormerLayer


def count_non_sequence_fields(feature_map):
    """Count context and target-item fields in QFormer token order."""
    num_fields = 0
    for feature, spec in feature_map.features.items():
        if feature in feature_map.labels or spec.get("type") == "meta":
            continue
        if spec.get("source") != "action":
            num_fields += 1
    return num_fields


def scale_normalized_field_keys(keys, field_head_log_scale):
    """Apply exp(alpha[field, head]) after QK normalization.

    Args:
        keys: normalized keys with shape [batch, head, field, head_dim].
        field_head_log_scale: alpha with shape [field, head].
    """
    num_heads = keys.size(1)
    num_fields = keys.size(2)
    expected_shape = (num_fields, num_heads)
    if tuple(field_head_log_scale.shape) != expected_shape:
        raise ValueError(
            "field_head_log_scale must have shape "
            f"{expected_shape}, got {tuple(field_head_log_scale.shape)}"
        )
    scale = field_head_log_scale.exp().transpose(0, 1)
    return keys * scale.view(1, num_heads, num_fields, 1)


class FieldAwareSDPACrossValueAttention(nn.Module):
    """V3 CrossValue with a field/head scale on normalized QK logits."""

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

    def forward(self, queries, keys, values, field_head_log_scale, mask=None):
        batch_size, num_queries, _ = queries.shape
        num_fields = keys.size(1)
        q = self.w_q(queries).view(
            batch_size, num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.w_k(keys).view(
            batch_size, num_fields, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.w_fi(values).view(
            batch_size, num_fields, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        k = scale_normalized_field_keys(k, field_head_log_scale)

        key_mask = None
        if mask is not None:
            key_mask = mask.to(dtype=torch.bool, device=q.device).view(
                batch_size, 1, 1, num_fields
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


class FieldAwareSDPAQFormerLayer(nn.Module):
    """A V3 layer whose NS CrossValue receives the shared field scale."""

    def __init__(self, token_dim, num_heads, ffn_dim, dropout=0.0,
                 qk_norm=False):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(token_dim)
        self.self_attn = SDPAMultiHeadAttention(
            token_dim, num_heads, qk_norm
        )
        self.cross_norm = nn.LayerNorm(token_dim)
        self.cross_attn = FieldAwareSDPACrossValueAttention(
            token_dim, num_heads, qk_norm
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = FeedForward(token_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, features, field_head_log_scale, mask=None):
        attn_out = self.self_attn(queries, queries, queries)
        queries = self.self_attn_norm(queries + self.dropout(attn_out))
        cross_out = self.cross_attn(
            queries,
            features,
            features,
            field_head_log_scale,
            mask,
        )
        queries = self.cross_norm(queries + self.dropout(cross_out))
        ffn_out = self.ffn(queries)
        return self.ffn_norm(queries + self.dropout(ffn_out))


class FieldAwareSDPAQFormerStage(nn.Module):
    """V3 NS QFormer with one shared alpha[field, head] parameter."""

    def __init__(self, token_dim, num_heads, num_layers, num_queries, ffn_dim,
                 num_fields, dropout=0.0, qk_norm=False):
        super().__init__()
        self.token_dim = token_dim
        self.num_queries = num_queries
        self.learnable_queries = nn.Parameter(
            torch.empty(num_queries, token_dim)
        )
        self.context_proj = nn.Linear(token_dim, token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.query_norm = nn.LayerNorm(token_dim)

        # The only additional trainable parameters: F x H, initialized so
        # exp(alpha) == 1 and the initial model is exactly the V3 baseline.
        self.field_head_log_scale = nn.Parameter(
            torch.zeros(num_fields, num_heads)
        )
        self.layers = nn.ModuleList([
            FieldAwareSDPAQFormerLayer(
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
        nn.init.zeros_(self.field_head_log_scale)

    def forward(self, features, mask=None):
        batch_size = features.size(0)
        if mask is None:
            pooled_features = features.mean(dim=1)
        else:
            valid = mask.to(
                dtype=features.dtype, device=features.device
            ).unsqueeze(-1)
            pooled_features = (features * valid).sum(dim=1)
            pooled_features = pooled_features / valid.sum(dim=1).clamp_min(1.0)

        context = self.context_norm(self.context_proj(pooled_features))
        queries = self.learnable_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        queries = self.query_norm(queries + context.unsqueeze(1))
        for layer in self.layers:
            queries = layer(
                queries,
                features,
                self.field_head_log_scale,
                mask,
            )
        return self.final_norm(queries)


class QFormerCross31(QFormerCross2):
    """QFormerCross3 plus static field/head logit calibration in NS reads."""

    def __init__(self, feature_map, model_id="QFormerCross31", **kwargs):
        num_fields = count_non_sequence_fields(feature_map)
        ns_stage_cls = partial(
            FieldAwareSDPAQFormerStage,
            num_fields=num_fields,
        )
        super().__init__(
            feature_map,
            model_id=model_id,
            _qformer_stage_cls=ns_stage_cls,
            _qformer_layer_cls=SDPAQFormerLayer,
            **kwargs,
        )
        self.ns_qformer.field_names = tuple(self.ns_features)

