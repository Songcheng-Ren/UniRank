# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
import torch.nn.functional as F
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, MultiHeadTargetAttention
from unirank.utils import not_in_whitelist


class SSR(MultiTaskModel):
    """SSR-D: Explicit Sparsity for Scalable Recommendation (Dynamic Instantiation).
    Multi-view Sparse Filtering directly operates on the concatenated feature
    embeddings, acting as the tokenization that decomposes the input into b
    parallel views for dimension-level sparse filtering (via ICS) followed by
    intra-view dense fusion.
    """

    def __init__(self,
                 feature_map,
                 model_id="SSR",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 attention_dropout=0,
                 embedding_dim=10,
                 num_layers=3,
                 num_iters=5,
                 init_alpha=0.1,
                 num_tasks=4,
                 token_dim=64,
                 attention_dim=None,
                 num_ns_token=4,
                 net_dropout=0,
                 accumulation_steps=1,
                 **kwargs):
        super(SSR, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  **kwargs)
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_ns_token = num_ns_token
        self.accumulation_steps = accumulation_steps

        # 统计非 item 特征维度、item 特征维度
        self.item_info_dim = 0
        self.non_item_dim = 0
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item", "action"]:
                self.item_info_dim += emb_dim
            else:
                self.non_item_dim += emb_dim

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        self.attention_layers = MultiHeadTargetAttention(
            input_dim=self.item_info_dim,
            attention_dim=token_dim if attention_dim is None else attention_dim,
            dropout_rate=attention_dropout
        )

        # SSR backbone: multi-view sparse filtering acts as tokenization on the
        # concatenated feature embeddings (d_in = non_item_dim + item_info_dim)
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim
        self.unified_interaction_layers = SSRBackbone(input_dim=input_dim,
                                          num_ns_token=num_ns_token,
                                          token_dim=token_dim,
                                          num_layers=num_layers,
                                          num_iters=num_iters,
                                          init_alpha=init_alpha,
                                          net_dropout=net_dropout)

        self.tower = nn.ModuleList([MLP_Block(input_dim=token_dim,
                                              output_dim=1,
                                              hidden_units=tower_hidden_units,
                                              hidden_activations=tower_activations,
                                              output_activation=None,
                                              dropout_rates=net_dropout)
                                    for _ in range(num_tasks)])
        if isinstance(task, list):
            assert len(task) == num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]
            )

        self.compile(kwargs.get("dense_optimizer"), kwargs["loss"], kwargs.get("dense_learning_rate"))
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_seq_emb[:, 0:-1, :]  # B x T x item_info_dim

        seq_pooling_emb = self.attention_layers(target_emb, sequence_emb, mask)  # B x embedding_dim

        # concat 所有 feature embeddings 作为 SSR backbone 的输入
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)       # B x non_item_dim
        feature_embeddings = torch.cat([user_context_emb, target_emb, seq_pooling_emb], dim=-1)  # B x d_in

        # Multi-view Sparse Filtering 直接作用于 concat 后的 feature embeddings，
        # 充当 tokenization 的角色，无需额外 tokenizer 包装
        bottom_output = self.activation_checkpoint(
            self.unified_interaction_layers,
            feature_embeddings
        )  # B x token_dim

        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict


class SSRBackbone(nn.Module):
    """Stack of SSR blocks. The first block takes the raw concatenated feature
    embeddings (d_in) as input — multi-view sparse filtering itself serves as
    the tokenization directly. Intermediate
    blocks keep concat aggregation, the final block switches to averaging.
    """

    def __init__(self,
                 input_dim,
                 num_ns_token,
                 token_dim,
                 num_layers=3,
                 num_iters=5,
                 init_alpha=0.1,
                 net_dropout=0.0):
        super(SSRBackbone, self).__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            # first block: input_dim = d_in; later blocks: input_dim = b*token_dim
            block_in = input_dim if i == 0 else num_ns_token * token_dim
            self.blocks.append(SSRBlock(
                input_dim=block_in,
                num_ns_token=num_ns_token,
                token_dim=token_dim,
                num_iters=num_iters,
                init_alpha=init_alpha,
                is_last=is_last,
                net_dropout=net_dropout,
            ))

    def forward(self, x):
        """
        x: B x input_dim  (concatenated feature embeddings)
        return: B x token_dim  (after last-layer averaging)
        """
        for block in self.blocks:
            x = block(x)
        return x  # B x token_dim


class SSRBlock(nn.Module):
    """One SSR layer with multi-view sparse filtering + intra-view dense fusion.

    Multi-view Sparse Filtering acts as the tokenization: it takes the
    high-dimensional input x (B x d_in) and produces b view tokens, each of
    dimension token_dim, via per-view projection + ICS sparse filtering.
    Then each view independently performs dense fusion.

    Intermediate layers aggregate views via concat(LN(z_1),...,LN(z_b)).
    The last layer aggregates views via mean(LN(z_i)) (see Eq. 8).
    """

    def __init__(self,
                 input_dim,
                 num_ns_token,
                 token_dim,
                 num_iters=5,
                 init_alpha=0.1,
                 is_last=False,
                 net_dropout=0.0):
        super(SSRBlock, self).__init__()
        self.num_ns_token = num_ns_token
        self.token_dim = token_dim
        self.is_last = is_last

        # per-view tokenization: x (d_in) -> z (token_dim), part of sparse filtering F_i
        self.proj = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                hidden_units=[token_dim],
                hidden_activations="SiLU",
                layer_norm=True,
            )
            for _ in range(num_ns_token)
        ])
        # per-view sparse filter (ICS): dynamic instantiation of F_i
        self.filters = nn.ModuleList([
            IterativeCompetitiveSparse(token_dim, num_iters=num_iters, init_alpha=init_alpha)
            for _ in range(num_ns_token)
        ])
        # per-view dense fusion M_i: z_i = act(h_i V_i + bias_i)  (Eq. 6)
        self.fusion = nn.ModuleList([
            nn.Linear(token_dim, token_dim)
            for _ in range(num_ns_token)
        ])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()

        # LayerNorm for aggregation; output dim = token_dim (last) or b*token_dim (intermediate)
        out_dim = token_dim if is_last else num_ns_token * token_dim
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: B x input_dim  (concatenated feature embeddings or previous layer output)
        return: B x token_dim              if is_last  (after mean aggregation)
                B x (num_ns_token*token_dim) otherwise  (after concat aggregation)
        """
        view_outputs = []
        for i in range(self.num_ns_token):
            z = self.proj[i](x)                 # B x token_dim  (project into view subspace)
            h = self.filters[i](z)              # B x token_dim  (sparse filtered via ICS)
            out = self.act(self.fusion[i](h))   # B x token_dim  (intra-view dense fusion)
            out = self.dropout(out)
            view_outputs.append(out)

        if self.is_last:
            # last-layer aggregation: mean of LayerNorm(z_i)  (Eq. 8)
            stacked = torch.stack([self.out_norm(v) for v in view_outputs], dim=1)  # B x b x token_dim
            return stacked.mean(dim=1)                                               # B x token_dim
        else:
            # intermediate aggregation: concat(LN(z_1),...,LN(z_b))  (Eq. 7)
            return self.out_norm(torch.cat(view_outputs, dim=-1))                    # B x (b*token_dim)


class IterativeCompetitiveSparse(nn.Module):
    """Iterative Competitive Sparse (ICS) operator for SSR-D.

    Implements Algorithm 1 of the paper:
      1. Initialize:  x(0) = ReLU(z)
      2. For t = 0..T-1:
           mu(t) = mean(x(t))
           x(t+1) = ReLU(x(t) - alpha_t * mu(t))
      3. Signal recovery: y = gamma * x(T)

    Here alpha_t (T learnable extinction rates) and gamma (learnable
    per-dimension scale) are optimized end-to-end.
    """

    def __init__(self, dim, num_iters=5, init_alpha=0.1):
        super(IterativeCompetitiveSparse, self).__init__()
        self.dim = dim
        self.num_iters = num_iters
        # T learnable extinction rates (one per iteration)
        self.alpha = nn.Parameter(torch.full((num_iters,), init_alpha))
        # learnable per-dimension rescaling for signal recovery
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, z):
        """
        z: B x dim  (projected feature of a single view)
        return: B x dim  (sparse-filtered feature)
        """
        x = F.relu(z)
        for t in range(self.num_iters):
            mu = x.mean(dim=-1, keepdim=True)          # B x 1
            x = F.relu(x - self.alpha[t] * mu)          # survival-of-the-fittest
        y = self.gamma * x                               # signal recovery
        return y
