# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# =========================================================================

import math

import torch
from torch import nn
import torch.nn.functional as F

from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block


class HSTU(MultiTaskModel):
    """Original dense HSTU-style ranker for UniRank's ranking inputs.

    This implementation follows the core HSTU equations:
      U,V,Q,K = split(SiLU(f1(X)))
      AV = phi2(QK^T + relative_bias) @ V
      Y = f2(LayerNorm(AV) * SiLU(U))

    Unlike UltraHSTU, it intentionally does not use FlexAttention, sparse
    local/recent windows, or intermediate sequence truncation.
    """

    def __init__(self,
                 feature_map,
                 model_id="HSTU",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=64,
                 num_heads=2,
                 net_dropout=0,
                 max_len=100,
                 accumulation_steps=1,
                 **kwargs):
        super(HSTU, self).__init__(feature_map, model_id=model_id, gpu=gpu, **kwargs)
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")

        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.accumulation_steps = accumulation_steps
        self.max_len = int(max_len)

        self.item_info_dim = 0
        self.user_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels or spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") == "item":
                self.item_info_dim += emb_dim
            elif spec.get("source") != "action":
                self.user_info_dim += emb_dim

        # Keep action as a separate embedding and add it to item-side tokens.
        self.feature_embedding_layer = FeatureEmbedding(
            feature_map, embedding_dim, not_required_feature_columns=["action"]
        )
        self.action_embedding_layer = FeatureEmbedding(
            feature_map, self.item_info_dim, required_feature_columns=["action"]
        )
        self.user_token = MLP_Block(
            input_dim=self.user_info_dim,
            hidden_units=[token_dim],
            hidden_activations="SiLU",
            layer_norm=True,
        )
        self.item_token = MLP_Block(
            input_dim=self.item_info_dim,
            hidden_units=[token_dim],
            hidden_activations="SiLU",
            layer_norm=True,
        )

        self.hstu_layers = HSTUStack(
            token_dim=token_dim,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            max_seq_len=self.max_len + 2,
            num_layers=num_layers,
            net_dropout=net_dropout,
        )

        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=token_dim,
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

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.size(0)

        item_emb = self.feature_embedding_layer(item_dict, flatten_emb=True)
        action_emb = self.action_embedding_layer(item_dict, flatten_emb=True)
        sequence_emb = (item_emb + action_emb).view(batch_size, -1, self.item_info_dim)

        user_context_emb = self.feature_embedding_layer(batch_dict, flatten_emb=True)
        user_token = self.user_token(user_context_emb).unsqueeze(1)
        sequence_tokens = self.item_token(sequence_emb)
        x = torch.cat([user_token, sequence_tokens], dim=1)

        # mask describes history only; candidate and user tokens are always valid.
        valid_mask = torch.cat([
            torch.ones(batch_size, 1, device=x.device, dtype=mask.dtype),
            mask.to(x.device),
            torch.ones(batch_size, 1, device=x.device, dtype=mask.dtype),
        ], dim=1)

        x = self.activation_checkpoint(self.hstu_layers, x, valid_mask)

        final_token = x[:, -1, :]
        tower_output = [self.tower[i](final_token) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        labels = self.feature_map.labels
        return {f"{labels[i]}_pred": y_pred[i] for i in range(self.num_tasks)}


class HSTULayer(nn.Module):
    """Dense causal HSTU layer with pointwise aggregated attention."""

    def __init__(self, token_dim, num_heads, expansion_factor, max_seq_len, net_dropout=0.0):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.max_seq_len = max_seq_len

        # f1: expand with a SiLU nonlinearity, then project to produce U,V,Q,K.
        # `expansion_factor` controls the width of the hidden expansion.
        hidden_dim = token_dim * expansion_factor
        self.pre_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(net_dropout),
            nn.Linear(hidden_dim, token_dim * 4),
        )
        self.out_norm = nn.LayerNorm(token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)
        self.dropout = nn.Dropout(net_dropout)
        self.relative_bias = nn.Embedding(2 * max_seq_len - 1, num_heads)

    def forward(self, x, valid_mask=None):
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"HSTU sequence length {seq_len} exceeds configured max_seq_len "
                f"{self.max_seq_len}"
            )

        projected = F.silu(self.pre_proj(x))
        u, v, q, k = projected.chunk(4, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seq_len, device=x.device)
        relative = positions[:, None] - positions[None, :]
        relative = relative + self.max_seq_len - 1
        bias = self.relative_bias(relative).permute(2, 0, 1).unsqueeze(0)
        logits = logits + bias.to(dtype=logits.dtype)

        # User token only reads itself; every other token reads the user token
        # and its causal prefix. This preserves causal ranking semantics.
        # `tril` already encodes this: row 0 is a one-hot on col 0, row i>0
        # attends to cols 0..i.
        causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        logits = logits.masked_fill(~causal.view(1, 1, seq_len, seq_len), 0.0)

        if valid_mask is not None:
            key_valid = valid_mask.bool().view(batch_size, 1, 1, seq_len)
            logits = logits.masked_fill(~key_valid, 0.0)

        # Pointwise aggregated attention: no Softmax normalization.
        # `logits` were already zeroed at invalid keys above; since SiLU(0)=0,
        # those positions contribute nothing and no second mask is needed.
        weights = F.silu(logits)
        aggregated = torch.matmul(weights, v)
        aggregated = aggregated.transpose(1, 2).contiguous().view(batch_size, seq_len, self.token_dim)

        y = self.out_proj(self.dropout(self.out_norm(aggregated) * F.silu(u)))
        if valid_mask is not None:
            y = y * valid_mask.unsqueeze(-1).to(dtype=y.dtype)
        return y


class HSTUStack(nn.Module):
    """Stack of residual HSTU layers, wrapped for activation checkpointing.

    Mirrors UltraHSTU's ``UnifiedInteractionBlocks`` so that HSTU honors the
    ``gradient_checkpointing`` config instead of silently ignoring it.
    """

    def __init__(self, token_dim, num_heads, expansion_factor, max_seq_len,
                 num_layers, net_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            HSTULayer(
                token_dim=token_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                max_seq_len=max_seq_len,
                net_dropout=net_dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, valid_mask=None):
        for layer in self.layers:
            x = layer(x, valid_mask=valid_mask) + x
        return x
