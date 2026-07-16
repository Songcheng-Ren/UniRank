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
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, MaskedAveragePooling, ScaledDotProductAttention
from unirank.utils import not_in_whitelist
from unirank.pytorch.layers.tokenization import AutoSplitTokenizer, FieldWiseTokenizer


class INFNet(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="INFNet",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=64,
                 num_c_hub_token=4,
                 num_t_hub_token=2,
                 net_dropout=0,
                 accumulation_steps=1,
                 **kwargs):
        super(INFNet, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     **kwargs)
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_c_hub_token = num_c_hub_token
        self.num_t_shared_hub = num_t_hub_token
        self.num_s_hub_token = num_tasks  # Each behavior type in the paper has a hub. Here, the types are divided by tasks.
        self.accumulation_steps = accumulation_steps
        self.num_field = feature_map.get_num_fields()

        # Track item and non-item feature dimensions
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

        # 1. Categorical Tokenization
        self.unified_tokenizer_layer = INFNetCategoricalTokenizer(
            embedding_dim=embedding_dim,
            token_dim=token_dim,
            num_c_hub_token=num_c_hub_token,
            num_field=self.num_field,
            expansion_factor=expansion_factor
        )

        # 2. Task Tokenization (Specific & Shared)
        self.task_orgs = nn.init.xavier_uniform_(nn.Parameter(torch.empty(1, self.num_tasks, token_dim)))
        self.task_shared_hubs = nn.init.xavier_uniform_(nn.Parameter(torch.empty(1, self.num_t_shared_hub, token_dim)))

        # 3. Sequence Tokenization
        self.seq_org_proj = nn.Linear(self.item_info_dim, token_dim)
        self.seq_hub_projs = nn.ModuleList([
            nn.Linear(self.item_info_dim, token_dim) for _ in range(self.num_tasks)
        ])
        self.masked_avg_pooling = MaskedAveragePooling()

        # INFNet Stacked Blocks (pass in num_layers to stack layer by layer)
        self.unified_interaction_layers = INFNetBlocks(
            num_layers=num_layers,
            token_dim=token_dim,
            num_c_hubs=self.num_c_hub_token,
            num_s_hubs=self.num_s_hub_token,
            num_t_hubs=self.num_tasks + self.num_t_shared_hub,
            expansion_factor=expansion_factor)

        # Task Towers
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
        batch_dict, item_dict, mask, multi_masks = self.get_inputs(inputs, return_multi_masks=True)
        batch_size = mask.shape[0]

        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]  # B x item_info_dim
        seq_orgs_raw = item_seq_emb[:, 0:-1, :]  # B x S x item_info_dim

        # --- Group 1: Sequence Tokens ---
        s_orgs = self.seq_org_proj(seq_orgs_raw)  # B x S x token_dim
        s_hubs = []
        for i in range(self.num_tasks):
            hub_i = self.masked_avg_pooling(self.seq_hub_projs[i](seq_orgs_raw), multi_masks[i])
            s_hubs.append(hub_i)
        s_hubs = torch.stack(s_hubs, dim=1)  # B x num_tasks x token_dim

        # --- Group 2: Categorical Tokens ---
        ns_emb = self.embedding_layer(batch_dict, flatten_emb=False)
        c_orgs_raw = torch.cat([ns_emb, target_emb.view(batch_size, -1, self.embedding_dim)], dim=1)
        c_orgs, c_hubs = self.unified_tokenizer_layer(c_orgs_raw)

        # --- Group 3: Task Tokens ---
        t_orgs = self.task_orgs.expand(batch_size, -1, -1)
        t_shared = self.task_shared_hubs.expand(batch_size, -1, -1)

        # Update Tokens and Hubs through multi-layer stacked INFNetBlocks
        t_orgs = self.activation_checkpoint(
            self.unified_interaction_layers,
            c_orgs,
            s_orgs,
            t_orgs,
            c_hubs,
            s_hubs,
            t_shared,
            mask
        )

        # --- Multi-Task Prediction ---
        tower_output = [self.tower[i](t_orgs[:, i, :]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict


class INFNetCategoricalTokenizer(nn.Module):
    def __init__(self, embedding_dim, token_dim, num_c_hub_token, num_field, expansion_factor=4):
        super(INFNetCategoricalTokenizer, self).__init__()
        input_dim = num_field * embedding_dim
        self.c_hub_mlps = AutoSplitTokenizer(input_dim, token_dim, num_c_hub_token)
        self.c_org_mlps = FieldWiseTokenizer(embedding_dim, token_dim, num_field)

    def forward(self, c_orgs_raw):
        B = c_orgs_raw.size(0)
        c_hubs_flat = c_orgs_raw.view(B, -1)
        c_orgs = self.c_org_mlps(c_orgs_raw)
        c_hubs = self.c_hub_mlps(c_hubs_flat)
        return c_orgs, c_hubs


class INFNetBlocks(nn.Module):
    """INFNet core module that manages multi-layer stacking: including Phase 1 (Aggregation) and Phase 2 (Broadcast)"""

    def __init__(self, num_layers, token_dim, num_c_hubs, num_s_hubs, num_t_hubs, expansion_factor=4):
        super(INFNetBlocks, self).__init__()
        self.num_layers = num_layers

        # Use nn.ModuleList to stack multiple layers of Aggregation and Broadcast respectively
        self.aggregation_layers = nn.ModuleList([
            Aggregation(token_dim) for _ in range(num_layers)
        ])
        self.broadcast_layers = nn.ModuleList([
            Broadcast(token_dim, num_c_hubs, num_s_hubs, num_t_hubs, expansion_factor)
            for _ in range(num_layers)
        ])

    def forward(self, c_orgs, s_orgs, t_orgs, c_hubs, s_hubs, t_shared, s_mask=None):
        # Equation (4): Task Hubs are composed of Specific and Shared
        t_hubs = torch.cat([t_orgs, t_shared], dim=1)
        # Iterative update layer by layer
        for i in range(self.num_layers):
            # Phase 1: Multi-View Global Aggregation
            c_hubs, s_hubs, t_hubs = self.aggregation_layers[i](
                c_hubs, s_hubs, t_hubs, c_orgs, s_orgs, t_orgs, s_mask
            )

            # Phase 2: Global-to-Local Affine Broadcast
            c_orgs, s_orgs, t_orgs = self.broadcast_layers[i](
                c_orgs, s_orgs, t_orgs, c_hubs, s_hubs, t_hubs
            )

        return t_orgs


class Aggregation(nn.Module):
    """ Multi-View Global Aggregation
    Keep SDPA mask-free to preserve FlashAttention eligibility, accepting minor attention leakage.
    """
    def __init__(self, token_dim):
        super(Aggregation, self).__init__()
        # Shared Keys/Values projection to reduce calculations
        self.k_proj_c = nn.Linear(token_dim, token_dim)
        self.v_proj_c = nn.Linear(token_dim, token_dim)
        self.k_proj_s = nn.Linear(token_dim, token_dim)
        self.v_proj_s = nn.Linear(token_dim, token_dim)
        self.k_proj_t = nn.Linear(token_dim, token_dim)
        self.v_proj_t = nn.Linear(token_dim, token_dim)

        # Independent Hub aggregator for each group
        self.c_agg = HubSpecificAggregator(token_dim)
        self.s_agg = HubSpecificAggregator(token_dim)
        self.t_agg = HubSpecificAggregator(token_dim)

    def forward(self, c_hubs, s_hubs, t_hubs, c_orgs, s_orgs, t_orgs, s_mask=None):
        K_c, V_c = self.k_proj_c(c_orgs), self.v_proj_c(c_orgs)
        K_s, V_s = self.k_proj_s(s_orgs), self.v_proj_s(s_orgs)
        K_t, V_t = self.k_proj_t(t_orgs), self.v_proj_t(t_orgs)

        # Processing sequence Mask broadcast dimensions (B, 1, T)
        if s_mask is not None:
            K_s = K_s * s_mask.unsqueeze(-1).float()
            V_s = V_s * s_mask.unsqueeze(-1).float()

        c_hubs_out = self.c_agg(c_hubs, K_c, V_c, K_s, V_s, K_t, V_t)
        s_hubs_out = self.s_agg(s_hubs, K_c, V_c, K_s, V_s, K_t, V_t)
        t_hubs_out = self.t_agg(t_hubs, K_c, V_c, K_s, V_s, K_t, V_t)

        return c_hubs_out, s_hubs_out, t_hubs_out

class HubSpecificAggregator(nn.Module):
    # Keep SDPA mask-free to preserve FlashAttention eligibility, accepting minor attention leakage.
    def __init__(self, token_dim):
        super(HubSpecificAggregator, self).__init__()
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.fuse = nn.Linear(token_dim * 3, token_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.dot_attention = ScaledDotProductAttention()

    def forward(self, hubs, K_c, V_c, K_s, V_s, K_t, V_t):
        Q = self.q_proj(hubs)
        scale = Q.size(-1) ** 0.5
        Z_c, _ = self.dot_attention(Q, K_c, V_c, scale=scale)
        Z_s, _ = self.dot_attention(Q, K_s, V_s, scale=scale)
        Z_t, _ = self.dot_attention(Q, K_t, V_t, scale=scale)
        Z_fused = self.fuse(torch.cat([Z_c, Z_s, Z_t], dim=-1))
        return self.norm(hubs + Z_fused)

class Broadcast(nn.Module):
    """Implements Section 3.2.2: Global-to-Local Affine Broadcast"""

    def __init__(self, token_dim, num_c_hubs, num_s_hubs, num_t_hubs, expansion_factor=4):
        super(Broadcast, self).__init__()
        self.c_bgu = BroadcastGatedUnit(token_dim, num_c_hubs, expansion_factor)
        self.s_bgu = BroadcastGatedUnit(token_dim, num_s_hubs, expansion_factor)
        self.t_bgu = BroadcastGatedUnit(token_dim, num_t_hubs, expansion_factor)

    def forward(self, c_orgs, s_orgs, t_orgs, c_hubs, s_hubs, t_hubs):
        c_orgs_out = self.c_bgu(c_orgs, c_hubs)
        s_orgs_out = self.s_bgu(s_orgs, s_hubs)
        t_orgs_out = self.t_bgu(t_orgs, t_hubs)
        return c_orgs_out, s_orgs_out, t_orgs_out


class BroadcastGatedUnit(nn.Module):
    """The BGU (Broadcast Gated Unit) mechanism in the paper"""

    def __init__(self, token_dim, num_hubs, expansion_factor=4):
        super(BroadcastGatedUnit, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_hubs * token_dim, token_dim * expansion_factor),
            nn.ReLU(),
            nn.Linear(token_dim * expansion_factor, token_dim * 2)
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, orgs, hubs):
        B = hubs.size(0)
        hubs_flat = hubs.view(B, -1)

        hyper_params = self.mlp(hubs_flat)
        alpha, beta = hyper_params.chunk(2, dim=-1)
        alpha = alpha.unsqueeze(1)  # B x 1 x token_dim
        beta = beta.unsqueeze(1)  # B x 1 x token_dim

        bgu_out = orgs * torch.sigmoid(alpha) + beta
        return self.norm(orgs + bgu_out)
