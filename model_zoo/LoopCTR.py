# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# =========================================================================

"""A minimal LoopCTR implementation with standard residual connections.

This variant preserves the defining LoopCTR ingredients while deliberately
leaving out HCR and MoE:

* an Entry -> shared Loop Block -> Exit sandwich;
* grouped global self-attention over user/context and target-item fields;
* prefix attention that prevents sequence tokens from reading target/global
  tokens while allowing global tokens to read the complete prefix;
* uniform process supervision over depths 0..L during training;
* configurable zero-/few-loop inference;
* Pre-LayerNorm residual blocks and SwiGLU feed-forward networks.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from unirank.pytorch.layers import FeatureEmbedding, MLP_Block
from unirank.pytorch.models import MultiTaskModel


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network used by all Transformer blocks."""

    def __init__(self, token_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(token_dim, hidden_dim)
        self.up_proj = nn.Linear(token_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        hidden = F.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        return self.dropout(self.down_proj(hidden))


class SDPAMultiHeadAttention(nn.Module):
    """Multi-head attention backed by PyTorch scaled-dot-product attention."""

    def __init__(self, token_dim, num_heads, attention_dropout=0.0,
                 qk_norm=False):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.attention_dropout = float(attention_dropout)

        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)
        self.q_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        )

    def _split_heads(self, inputs):
        batch_size, length, _ = inputs.shape
        return inputs.view(
            batch_size, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, query, key, value, key_mask=None, query_mask=None):
        batch_size = query.size(0)
        q = self.q_norm(self._split_heads(self.q_proj(query)))
        k = self.k_norm(self._split_heads(self.k_proj(key)))
        v = self._split_heads(self.v_proj(value))

        attention_mask = None
        has_valid_key = None
        if key_mask is not None:
            valid_key = key_mask.to(dtype=torch.bool, device=q.device)
            attention_mask = valid_key.view(
                batch_size, 1, 1, key.size(1)
            )
            has_valid_key = valid_key.any(dim=1).view(batch_size, 1, 1)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query.size(1), self.token_dim
        )
        output = self.out_proj(output)

        # Explicitly zero all-empty key rows and padded query positions. This
        # keeps empty histories finite across PyTorch SDPA implementations.
        if has_valid_key is not None:
            output = output * has_valid_key.to(dtype=output.dtype)
        if query_mask is not None:
            output = output * query_mask.to(
                dtype=output.dtype, device=output.device
            ).unsqueeze(-1)
        return output


class TransformerBlock(nn.Module):
    """Standard Pre-LN attention/SwiGLU block with ordinary residuals."""

    def __init__(self, token_dim, num_heads, ffn_dim, dropout=0.0,
                 attention_dropout=0.0, qk_norm=False):
        super().__init__()
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = SDPAMultiHeadAttention(
            token_dim=token_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            qk_norm=qk_norm,
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = SwiGLU(token_dim, ffn_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, tokens, mask=None):
        normalized = self.attention_norm(tokens)
        delta = self.attention(
            normalized,
            normalized,
            normalized,
            key_mask=mask,
            query_mask=mask,
        )
        tokens = tokens + self.residual_dropout(delta)
        tokens = tokens + self.residual_dropout(
            self.ffn(self.ffn_norm(tokens))
        )
        if mask is not None:
            tokens = tokens * mask.to(
                dtype=tokens.dtype, device=tokens.device
            ).unsqueeze(-1)
        return tokens


class EntryBlock(nn.Module):
    """Encode the sequence and configured global groups independently."""

    def __init__(self, token_dim, num_heads, ffn_dim, dropout=0.0,
                 attention_dropout=0.0, qk_norm=False,
                 global_group_sizes=()):
        super().__init__()
        if not global_group_sizes or any(
            size < 1 for size in global_group_sizes
        ):
            raise ValueError("global_group_sizes must contain positive sizes")
        self.global_group_sizes = tuple(global_group_sizes)
        self.sequence_block = TransformerBlock(
            token_dim, num_heads, ffn_dim, dropout,
            attention_dropout, qk_norm,
        )
        self.global_block = TransformerBlock(
            token_dim, num_heads, ffn_dim, dropout,
            attention_dropout, qk_norm,
        )

    def forward(self, sequence_tokens, global_tokens, sequence_mask):
        sequence_tokens = self.sequence_block(
            sequence_tokens, sequence_mask
        )

        batch_size, num_fields, token_dim = global_tokens.shape
        if sum(self.global_group_sizes) != num_fields:
            raise ValueError(
                "global group sizes do not match the number of global tokens"
            )

        if all(size == 1 for size in self.global_group_sizes):
            # Keep the paper's singleton-field mode efficient by folding all
            # fields into the batch for a single attention call.
            singleton_fields = global_tokens.reshape(
                batch_size * num_fields, 1, token_dim
            )
            singleton_fields = self.global_block(singleton_fields)
            global_tokens = singleton_fields.reshape(
                batch_size, num_fields, token_dim
            )
        else:
            global_groups = torch.split(
                global_tokens, self.global_group_sizes, dim=1
            )
            global_tokens = torch.cat([
                self.global_block(group) for group in global_groups
            ], dim=1)
        return sequence_tokens, global_tokens


class PrefixAttention(nn.Module):
    """LoopCTR prefix attention with shared projections for both streams."""

    def __init__(self, token_dim, num_heads, attention_dropout=0.0,
                 qk_norm=False):
        super().__init__()
        self.attention = SDPAMultiHeadAttention(
            token_dim=token_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            qk_norm=qk_norm,
        )

    def forward(self, sequence_tokens, global_tokens, sequence_mask):
        # Sequence representations remain target independent and cacheable.
        sequence_delta = self.attention(
            sequence_tokens,
            sequence_tokens,
            sequence_tokens,
            key_mask=sequence_mask,
            query_mask=sequence_mask,
        )

        prefix = torch.cat([sequence_tokens, global_tokens], dim=1)
        global_mask = torch.ones(
            global_tokens.shape[:2],
            dtype=sequence_mask.dtype,
            device=sequence_mask.device,
        )
        prefix_mask = torch.cat([sequence_mask, global_mask], dim=1)
        global_delta = self.attention(
            global_tokens,
            prefix,
            prefix,
            key_mask=prefix_mask,
        )
        return sequence_delta, global_delta


class LoopBlock(nn.Module):
    """The single shared recurrent block used for every loop iteration."""

    def __init__(self, token_dim, num_heads, ffn_dim, dropout=0.0,
                 attention_dropout=0.0, qk_norm=False):
        super().__init__()
        self.attention_norm = nn.LayerNorm(token_dim)
        self.prefix_attention = PrefixAttention(
            token_dim, num_heads, attention_dropout, qk_norm
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = SwiGLU(token_dim, ffn_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, sequence_tokens, global_tokens, sequence_mask):
        normalized_sequence = self.attention_norm(sequence_tokens)
        normalized_global = self.attention_norm(global_tokens)
        sequence_delta, global_delta = self.prefix_attention(
            normalized_sequence,
            normalized_global,
            sequence_mask,
        )
        sequence_tokens = sequence_tokens + self.residual_dropout(
            sequence_delta
        )
        global_tokens = global_tokens + self.residual_dropout(global_delta)

        sequence_tokens = sequence_tokens + self.residual_dropout(
            self.ffn(self.ffn_norm(sequence_tokens))
        )
        global_tokens = global_tokens + self.residual_dropout(
            self.ffn(self.ffn_norm(global_tokens))
        )
        sequence_tokens = sequence_tokens * sequence_mask.to(
            dtype=sequence_tokens.dtype, device=sequence_tokens.device
        ).unsqueeze(-1)
        return sequence_tokens, global_tokens


class ExitBlock(nn.Module):
    """Let global fields read the sequence before the prediction towers."""

    def __init__(self, token_dim, num_heads, ffn_dim, dropout=0.0,
                 attention_dropout=0.0, qk_norm=False):
        super().__init__()
        self.query_norm = nn.LayerNorm(token_dim)
        self.sequence_norm = nn.LayerNorm(token_dim)
        self.cross_attention = SDPAMultiHeadAttention(
            token_dim=token_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            qk_norm=qk_norm,
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = SwiGLU(token_dim, ffn_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, sequence_tokens, global_tokens, sequence_mask):
        delta = self.cross_attention(
            self.query_norm(global_tokens),
            self.sequence_norm(sequence_tokens),
            self.sequence_norm(sequence_tokens),
            key_mask=sequence_mask,
        )
        global_tokens = global_tokens + self.residual_dropout(delta)
        return global_tokens + self.residual_dropout(
            self.ffn(self.ffn_norm(global_tokens))
        )


class LoopCTR(MultiTaskModel):
    """LoopCTR without HCR or MoE, adapted to UniRank feature conventions."""

    def __init__(self,
                 feature_map,
                 model_id="LoopCTR",
                 task=("binary_classification",),
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=(256, 128),
                 embedding_dim=16,
                 token_dim=64,
                 num_heads=4,
                 num_loops=3,
                 inference_loops=1,
                 ffn_ratio=4.0,
                 qk_norm=False,
                 global_grouping="source",
                 max_len=100,
                 num_tasks=1,
                 attention_dropout=0.0,
                 net_dropout=0.0,
                 process_supervision=True,
                 accumulation_steps=1,
                 **kwargs):
        super().__init__(feature_map, model_id=model_id, gpu=gpu, **kwargs)
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if num_loops < 0:
            raise ValueError("num_loops must be non-negative")
        if not 0 <= inference_loops <= num_loops:
            raise ValueError(
                "inference_loops must be between 0 and num_loops"
            )
        if ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be positive")
        if max_len < 1:
            raise ValueError("max_len must be positive")

        self.feature_map = feature_map
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_loops = int(num_loops)
        self.inference_loops = int(inference_loops)
        self.process_supervision = bool(process_supervision)
        self.global_grouping = str(global_grouping).strip().lower()
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
                self.sequence_features.append(feature)
            else:
                self.context_features.append(feature)
        if not self.target_item_features:
            raise ValueError("LoopCTR requires target item features")
        if not self.sequence_features:
            raise ValueError("LoopCTR requires sequence features")

        self.global_features = (
            self.context_features + self.target_item_features
        )
        if self.global_grouping == "source":
            self.global_group_sizes = tuple(
                size for size in (
                    len(self.context_features),
                    len(self.target_item_features),
                )
                if size > 0
            )
        elif self.global_grouping == "all":
            self.global_group_sizes = (len(self.global_features),)
        elif self.global_grouping == "field":
            self.global_group_sizes = (1,) * len(self.global_features)
        else:
            raise ValueError(
                "global_grouping must be one of: source, all, field"
            )
        self.sequence_input_dim = sum(
            feature_map.features[feature].get(
                "embedding_dim", embedding_dim
            )
            for feature in self.sequence_features
        )

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.global_projection = nn.ModuleDict({
            feature: nn.Linear(
                feature_map.features[feature].get(
                    "embedding_dim", embedding_dim
                ),
                token_dim,
                bias=False,
            )
            for feature in self.global_features
        })
        self.global_field_embedding = nn.Parameter(torch.empty(
            len(self.global_features), token_dim
        ))
        self.global_input_norm = nn.LayerNorm(token_dim)

        self.sequence_projection = nn.Linear(
            self.sequence_input_dim, token_dim
        )
        self.position_embedding = nn.Parameter(torch.empty(max_len, token_dim))
        self.sequence_input_norm = nn.LayerNorm(token_dim)

        ffn_dim = int(token_dim * ffn_ratio)
        self.entry_block = EntryBlock(
            token_dim, num_heads, ffn_dim, net_dropout,
            attention_dropout, qk_norm,
            global_group_sizes=self.global_group_sizes,
        )
        # There is exactly one Loop Block. Forward invokes this same module L
        # times, so parameters are independent of num_loops.
        self.loop_block = LoopBlock(
            token_dim, num_heads, ffn_dim, net_dropout,
            attention_dropout, qk_norm,
        )
        self.exit_block = ExitBlock(
            token_dim, num_heads, ffn_dim, net_dropout,
            attention_dropout, qk_norm,
        )

        tower_input_dim = len(self.global_features) * token_dim
        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=tower_input_dim,
                output_dim=1,
                hidden_units=list(tower_hidden_units),
                hidden_activations=tower_activations,
                output_activation=None,
                dropout_rates=net_dropout,
            )
            for _ in range(num_tasks)
        ])
        if isinstance(task, (list, tuple)):
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
        nn.init.normal_(self.global_field_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        self.model_to_device()

    def set_inference_loops(self, inference_loops):
        """Change deployed compute depth without rebuilding the model."""
        if not 0 <= inference_loops <= self.num_loops:
            raise ValueError(
                "inference_loops must be between 0 and num_loops"
            )
        self.inference_loops = int(inference_loops)

    def _build_global_tokens(self, context_embeddings, item_embeddings):
        tokens = []
        for feature in self.context_features:
            embedding = context_embeddings[feature]
            if embedding.dim() != 2:
                raise ValueError(
                    f"context feature {feature} must produce B x D "
                    f"embeddings, got shape={tuple(embedding.shape)}"
                )
            tokens.append(self.global_projection[feature](embedding))
        for feature in self.target_item_features:
            embedding = item_embeddings[feature]
            if embedding.dim() != 3:
                raise ValueError(
                    f"item feature {feature} must produce B x (S+1) x D "
                    f"embeddings, got shape={tuple(embedding.shape)}"
                )
            tokens.append(
                self.global_projection[feature](embedding[:, -1, :])
            )
        global_tokens = torch.stack(tokens, dim=1)
        return self.global_input_norm(
            global_tokens + self.global_field_embedding.unsqueeze(0)
        )

    def _build_sequence_tokens(self, item_embeddings, sequence_mask):
        history_fields = [
            item_embeddings[feature][:, :-1, :]
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
        valid_mask = sequence_mask.to(dtype=torch.bool)
        position_ids = valid_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        sequence_tokens = self.sequence_input_norm(
            sequence_tokens + self.position_embedding[position_ids]
        )
        return sequence_tokens * valid_mask.unsqueeze(-1).to(
            dtype=sequence_tokens.dtype
        )

    def _predict(self, sequence_tokens, global_tokens, sequence_mask):
        exited_global = self.activation_checkpoint(
            self.exit_block,
            sequence_tokens,
            global_tokens,
            sequence_mask,
        )
        bottom_output = exited_global.flatten(start_dim=1)
        logits = [
            self.tower[index](bottom_output)
            for index in range(self.num_tasks)
        ]
        return [
            self.output_activation[index](logits[index])
            for index in range(self.num_tasks)
        ]

    def forward(self, inputs):
        batch_dict, item_dict, sequence_mask = self.get_inputs(inputs)
        if sequence_mask.size(1) > self.max_len:
            raise ValueError(
                f"input history length {sequence_mask.size(1)} exceeds "
                f"max_len={self.max_len}"
            )

        context_embeddings = self.embedding_layer.embedding_layer(batch_dict)
        item_embeddings = self.embedding_layer.embedding_layer(item_dict)
        global_tokens = self._build_global_tokens(
            context_embeddings, item_embeddings
        )
        sequence_mask = sequence_mask.to(device=global_tokens.device)
        sequence_tokens = self._build_sequence_tokens(
            item_embeddings, sequence_mask
        )
        sequence_tokens, global_tokens = self.activation_checkpoint(
            self.entry_block,
            sequence_tokens,
            global_tokens,
            sequence_mask,
        )

        collect_process_predictions = self.training and self.process_supervision
        executed_loops = (
            self.num_loops if self.training else self.inference_loops
        )
        depth_predictions = []
        if collect_process_predictions:
            depth_predictions.append(
                self._predict(sequence_tokens, global_tokens, sequence_mask)
            )

        for _ in range(executed_loops):
            sequence_tokens, global_tokens = self.activation_checkpoint(
                self.loop_block,
                sequence_tokens,
                global_tokens,
                sequence_mask,
            )
            if collect_process_predictions:
                depth_predictions.append(
                    self._predict(
                        sequence_tokens, global_tokens, sequence_mask
                    )
                )

        if collect_process_predictions:
            predictions = depth_predictions[-1]
        else:
            predictions = self._predict(
                sequence_tokens, global_tokens, sequence_mask
            )

        labels = self.feature_map.labels
        return_dict = {
            f"{labels[index]}_pred": predictions[index]
            for index in range(self.num_tasks)
        }
        for depth, depth_output in enumerate(depth_predictions):
            for index, label in enumerate(labels):
                return_dict[
                    f"{label}_pred_loop_{depth}"
                ] = depth_output[index]
        return return_dict

    def compute_loss(self, return_dict, y_true):
        if not (self.training and self.process_supervision):
            return self.add_loss(return_dict, y_true)

        labels = self.feature_map.labels
        depth_losses = []
        for depth in range(self.num_loops + 1):
            depth_dict = {
                f"{label}_pred": return_dict[
                    f"{label}_pred_loop_{depth}"
                ]
                for label in labels
            }
            depth_losses.append(self.add_loss(depth_dict, y_true))
        return torch.stack(depth_losses).mean()
