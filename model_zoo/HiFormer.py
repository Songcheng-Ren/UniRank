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
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, MaskedAveragePooling, PerTokenFeedForward, MultiHeadTargetAttention, ScaledDotProductAttention
from unirank.pytorch.torch_utils import get_activation
from unirank.utils import not_in_whitelist
from unirank.pytorch.layers.tokenization import FieldWiseTokenizer


class HiFormer(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="HiFormer",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 num_layers=3,
                 attention_dropout=0,
                 expansion_factor=4,
                 num_tasks=4,
                 token_dim=64,
                 attention_dim=None,
                 num_heads=2,
                 qkv_rank=256,
                 net_dropout=0,
                 accumulation_steps=1,
                 attention_activation_type="SoftMax",
                 **kwargs):
        super(HiFormer, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       **kwargs)
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
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
                self.num_field += 1
            else:
                self.non_item_dim += emb_dim

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.unified_tokenizer_layer = HiFormerTokenizer(
            input_dim=embedding_dim,
            token_dim=token_dim,
            num_tokens=self.num_field,
            num_tasks=num_tasks
        )

        self.attention_layers = MultiHeadTargetAttention(
            input_dim=self.item_info_dim,
            attention_dim=token_dim if attention_dim is None else attention_dim,
            dropout_rate=attention_dropout,
            attention_activation_type=attention_activation_type,
        )

        self.unified_interaction_layers = HiFormerBlocks(token_dim=token_dim,
                                       num_token=num_tasks + self.num_field,
                                       num_layers=num_layers,
                                       expand=expansion_factor,
                                       num_heads=num_heads,
                                       qkv_rank=qkv_rank,
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

        seq_pooling_emb = self.attention_layers(target_emb, sequence_emb, mask)

        # Other non-sequential features -> NS tokens
        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=False)
        feature_embeddings = torch.cat([user_context_emb,
                                        target_emb.view(batch_size, -1, self.embedding_dim),
                                        seq_pooling_emb.view(batch_size, -1, self.embedding_dim)], dim=1)
        unified_tokens = self.unified_tokenizer_layer(feature_embeddings)

        # unified model
        unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            unified_tokens
        )

        # use task tokens for task-specific towers
        task_tokens = unified_tokens[:, self.num_field:, :]  # [B, num_tasks, token_dim]
        tower_output = [self.tower[i](task_tokens[:, i, :]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict


class HiFormerTokenizer(nn.Module):
    def __init__(self, input_dim, token_dim, num_tokens, num_tasks):
        super(HiFormerTokenizer, self).__init__()
        self.task_tokens = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(1, num_tasks, token_dim))
        )
        self.token_mlps = FieldWiseTokenizer(input_dim, token_dim, num_tokens)

    def forward(self, feature_embeddings):
        tokens = self.token_mlps(feature_embeddings)
        task_tokens = self.task_tokens.expand(tokens.size(0), -1, -1)
        return torch.cat([tokens, task_tokens], dim=1)


class HiFormerBlocks(nn.Module):
    def __init__(self,
                 token_dim,
                 num_token,
                 num_layers,
                 expand=2,
                 num_heads=2,
                 qkv_rank=256,
                 net_dropout=0.0):
        super(HiFormerBlocks, self).__init__()
        self.num_layers = num_layers
        self.mixer_norms = nn.ModuleList([
            nn.LayerNorm(token_dim)
            for _ in range(num_layers)
        ])
        self.pffn_norms = nn.ModuleList([
            nn.LayerNorm(token_dim)
            for _ in range(num_layers)
        ])

        self.mixer_layers = nn.ModuleList([
            HiformerAttentionLayer(token_dim=token_dim,
                                        num_token=num_token,
                                        num_heads=num_heads,
                                        qkv_rank=qkv_rank)
            for _ in range(num_layers)
        ])
        self.pffn_layers = nn.ModuleList([
            PerTokenFeedForward(input_dim=token_dim,
                                num_token=num_token,
                                expand=expand,
                                net_dropout=net_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        for i in range(self.num_layers):
            x = self.mixer_layers[i](self.mixer_norms[i](x)) + x
            x = self.pffn_layers[i](self.pffn_norms[i](x)) + x
        return x

class LowRankLinear(nn.Module):
    def __init__(self, input_dim, output_dim, rank, bias=False):
        super(LowRankLinear, self).__init__()
        assert rank > 0, "rank must be positive"
        self.down_proj = nn.Linear(input_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, output_dim, bias=bias)

    def forward(self, x):
        return self.up_proj(self.down_proj(x))


class HiformerAttentionLayer(nn.Module):
    def __init__(self,
                 token_dim,
                 num_token,
                 num_heads=2,
                 qkv_rank=256):
        super(HiformerAttentionLayer, self).__init__()
        assert token_dim % num_heads == 0, "token_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.token_dim = token_dim
        self.num_token = num_token
        input_dim = num_token * token_dim
        self.W_q = LowRankLinear(input_dim, input_dim, qkv_rank, bias=False)
        self.W_k = LowRankLinear(input_dim, input_dim, qkv_rank, bias=False)
        self.W_v = LowRankLinear(input_dim, input_dim, qkv_rank, bias=False)
        self.dot_attention = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor): # B × T × D
        x = x.flatten(start_dim=1) # B × TD
        Q = self.W_q(x).view(-1, self.num_token, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, T, Dh]
        K = self.W_k(x).view(-1, self.num_token, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(-1, self.num_token, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out, _ = self.dot_attention(Q, K, V, scale=self.head_dim ** 0.5)
        attn_out = attn_out.transpose(1, 2).contiguous().view(-1, self.num_token, self.token_dim)  # [B, T, D]
        return attn_out
