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
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, MultiHeadTokenMixing, PerTokenSwiGLU, SwiGLU
from unirank.pytorch.torch_utils import get_activation
from unirank.utils import not_in_whitelist
from unirank.pytorch.layers.tokenization import ChunkTokenizer


class MixFormer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="MixFormer",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=64,
                 num_ns_token=4,
                 net_dropout=0,
                 accumulation_steps=1,
                 **kwargs):
        super(MixFormer, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       **kwargs)
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_ns_token = num_ns_token
        self.accumulation_steps = accumulation_steps

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
        # Non-sequential feature tokenizer: generates num_ns_token NS tokens, because the target item is also regarded as an NS token, so self.item_info_dim is added
        self.unified_tokenizer_layer = ChunkTokenizer(
            input_dim=self.non_item_dim + self.item_info_dim,
            token_dim=token_dim,
            num_tokens=num_ns_token
        )

        # Project sequence and target items to token_dim
        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        self.unified_interaction_layers = MixFormerBlocks(input_dim=token_dim,
                                         num_ns_token=self.num_ns_token,
                                         num_layers=num_layers,
                                         expand=expansion_factor,
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
        # item_dict contains [history_items..., target_item]
        # Reshape flattened embeddings to B x (T+1) x item_info_dim
        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_seq_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_seq_emb[:, 0:-1, :]  # B x T x item_info_dim

        # S-tokens
        s_tokens = self.item_token_proj(sequence_emb)  # B x T x token_dim

        # target item as an NS token
        # Other non-sequential features -> NS tokens
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)  # B x non_item_dim
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)
        unified_tokens = self.unified_tokenizer_layer(feature_embeddings)  # B x num_ns_token x token_dim

        # unified model
        _, unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            s_tokens,
            unified_tokens,
            mask
        )

        bottom_output = unified_tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict

class MixFormerBlocks(nn.Module):
    """
    Each layer applies:
    1) Query Mixer
    2) Cross Attention
    3) Output Fusion
    """
    def __init__(self,
                 input_dim,
                 num_ns_token,
                 num_layers,
                 expand=4,
                 net_dropout=0.0):
        super(MixFormerBlocks, self).__init__()
        self.num_layers = num_layers

        self.query_mixers = nn.ModuleList([
            QueryMixer(
                input_dim=input_dim,
                num_ns_token=num_ns_token,
                expand=expand,
                net_dropout=net_dropout
            ) for _ in range(num_layers)
        ])

        self.cross_attentions = nn.ModuleList([
            CrossAttention(
                input_dim=input_dim,
                num_ns_token=num_ns_token,
                expand=expand,
                net_dropout=net_dropout
            ) for _ in range(num_layers)
        ])

        self.output_fusions = nn.ModuleList([
            OutputFusion(
                input_dim=input_dim,
                num_ns_token=num_ns_token,
                expand=expand,
                net_dropout=net_dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, s_tokens, ns_tokens, mask=None):
        """
        s_tokens:  B x T x D
        ns_tokens: B x N x D
        mask:      B x T
        """
        for i in range(self.num_layers):
            ns_tokens = self.query_mixers[i](ns_tokens)                 # B x N x D
            s_tokens, ns_tokens = self.cross_attentions[i](s_tokens, ns_tokens, mask)
            ns_tokens = self.output_fusions[i](ns_tokens)               # B x N x D
        return s_tokens, ns_tokens

class QueryMixer(nn.Module):
    """
    Implements the paper's Query Mixer:
    P = HeadMixing(Norm(X)) + X
    q_i = PerHeadFFN(Norm(p_i)) + p_i
    """
    def __init__(self, input_dim, num_ns_token, expand=4, net_dropout=0.0):
        super(QueryMixer, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.head_mixing = MultiHeadTokenMixing(input_dim=input_dim, num_token=num_ns_token)
        self.per_head_ffn = PerTokenSwiGLU(input_dim=input_dim,
                                                num_token=num_ns_token,
                                                expand=expand,
                                                net_dropout=net_dropout)

    def forward(self, x):
        x = self.head_mixing(self.norm1(x)) + x
        x = self.per_head_ffn(self.norm2(x)) + x
        return x

class CrossAttention(nn.Module):
    """
    Implements the paper's Cross Attention:
    - First do per-layer FFN refinement on the sequence
    - Then each NS head performs cross attention on the sequence
    - Keep SDPA mask-free to preserve FlashAttention eligibility, accepting minor attention leakage.
    """
    def __init__(self, input_dim, num_ns_token, expand=4, net_dropout=0.0):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.num_ns_token = num_ns_token

        self.seq_norm = nn.LayerNorm(input_dim)

        self.seq_ffn = SwiGLU(
            input_dim=input_dim,
            expand=expand,
            net_dropout=net_dropout
        )

        # A set of K/V projections for each query head
        self.k_proj = nn.Linear(input_dim, input_dim * num_ns_token)
        self.v_proj = nn.Linear(input_dim, input_dim * num_ns_token)

    def forward(self, s_tokens, ns_tokens, mask=None):
        """
        s_tokens: B x T x D
        ns_tokens: B x N x D
        mask: B x T (1/True indicates valid position)
        """
        # per-layer sequence refinement
        s_tokens = self.seq_ffn(self.seq_norm(s_tokens)) + s_tokens  # B x T x D

        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()

        keys = self.k_proj(s_tokens).chunk(self.num_ns_token, dim=-1)
        values = self.v_proj(s_tokens).chunk(self.num_ns_token, dim=-1)
        k = torch.stack(keys, dim=1) # K/V: B x N x T x D
        v = torch.stack(values, dim=1)  # K/V: B x N x T x D
        q = ns_tokens.unsqueeze(2) # B × N x 1 x D

        # B x N x D
        ns_tokens = F.scaled_dot_product_attention(q, k, v).squeeze(2) + ns_tokens
        return s_tokens, ns_tokens

class OutputFusion(nn.Module):
    """
    Implements the paper's Output Fusion:
    o_i = PerHeadFFN(Norm(z_i)) + z_i
    """
    def __init__(self, input_dim, num_ns_token, expand=4, net_dropout=0.0):
        super(OutputFusion, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.per_head_ffn = PerTokenSwiGLU(input_dim=input_dim,
                                           num_token=num_ns_token,
                                           expand=expand,
                                           net_dropout=net_dropout)

    def forward(self, ns_tokens):
        ns_tokens = self.per_head_ffn(self.norm(ns_tokens)) + ns_tokens
        return ns_tokens
