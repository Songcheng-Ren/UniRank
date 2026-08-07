# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# =========================================================================

import torch
from torch import nn

from unirank.pytorch.layers import FeatureEmbedding, MLP_Block
from unirank.pytorch.models import MultiTaskModel


class DCNv2(MultiTaskModel):
    """DCNv2 ranking baseline adapted to UniRank's sequence-aware inputs.

    The cross input contains user/context fields, the candidate item, and a
    masked mean-pooled representation of the preceding item/action history.
    The cross branch uses the original full-matrix DCNv2 cross layers, while
    a parallel deep tower learns unconstrained feature transformations.
    """

    def __init__(self,
                 feature_map,
                 model_id="DCNv2",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[256, 128],
                 embedding_dim=16,
                 num_layers=3,
                 num_tasks=4,
                 net_dropout=0,
                 accumulation_steps=1,
                 **kwargs):
        super(DCNv2, self).__init__(feature_map, model_id=model_id, gpu=gpu, **kwargs)
        self.feature_map = feature_map
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.accumulation_steps = accumulation_steps

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.item_info_dim = 0
        self.non_item_dim = 0
        for feat, spec in feature_map.features.items():
            if feat in feature_map.labels or spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ("item", "action"):
                self.item_info_dim += emb_dim
            else:
                self.non_item_dim += emb_dim

        # user/context + candidate item + pooled history
        self.input_dim = self.non_item_dim + 2 * self.item_info_dim
        self.cross_net = CrossNet(
            input_dim=self.input_dim,
            num_layers=num_layers,
            dropout=net_dropout,
        )
        self.deep_net = MLP_Block(
            input_dim=self.input_dim,
            output_dim=None,
            hidden_units=tower_hidden_units,
            hidden_activations=tower_activations,
            output_activation=None,
            dropout_rates=net_dropout,
        )
        deep_dim = tower_hidden_units[-1] if tower_hidden_units else self.input_dim
        combined_dim = self.input_dim + deep_dim
        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=combined_dim,
                output_dim=1,
                hidden_units=[],
                # ``tower_activations`` only governs the deep tower above;
                # here hidden_units=[] means a single Linear, so no hidden
                # activation is needed for the final task head.
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

        # item_dict is [history..., candidate]. Embed all item-side fields.
        item_embeddings = self.embedding_layer(item_dict, flatten_emb=True)
        item_embeddings = item_embeddings.view(batch_size, -1, self.item_info_dim)
        history_embeddings = item_embeddings[:, :-1, :]
        candidate_embedding = item_embeddings[:, -1, :]

        history_mask = mask.to(dtype=history_embeddings.dtype, device=history_embeddings.device)
        history_sum = (history_embeddings * history_mask.unsqueeze(-1)).sum(dim=1)
        history_count = history_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        history_embedding = history_sum / history_count

        context_embedding = self.embedding_layer(batch_dict, flatten_emb=True)
        x0 = torch.cat([context_embedding, candidate_embedding, history_embedding], dim=-1)

        cross_output = self.cross_net(x0)
        deep_output = self.deep_net(x0)
        combined = torch.cat([cross_output, deep_output], dim=-1)
        tower_output = [self.tower[i](combined) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        labels = self.feature_map.labels
        return {f"{labels[i]}_pred": y_pred[i] for i in range(self.num_tasks)}


class CrossNet(nn.Module):
    """Original full-matrix DCNv2 cross network.

    Each layer applies x_{l+1} = x0 * (W_l x_l + b_l) + x_l.
    There are no mixture gates, experts, or low-rank factorization here.
    """

    def __init__(self, input_dim, num_layers=3, dropout=0.0):
        super().__init__()
        if input_dim <= 0 or num_layers <= 0:
            raise ValueError("input_dim and num_layers must be positive")
        self.input_dim = input_dim
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, input_dim))
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x0):
        x = x0
        for weight, bias in zip(self.weights, self.biases):
            cross = torch.matmul(x, weight) + bias
            cross = self.dropout(cross)
            x = x0 * cross + x
        return x

