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
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, MultiHeadTargetAttention, PerTokenSwiGLU, ScaledDotProductAttention
from unirank.pytorch.torch_utils import get_activation
from unirank.utils import not_in_whitelist
from unirank.pytorch.layers.tokenization import ChunkTokenizer


class Zenith(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="Zenith",
                 task=["binary_classification"],
                 gpu=-1,
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 attention_dropout=0,
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_tasks=4,
                 id_dim=64,
                 token_dim=64,
                 attention_dim=None,
                 num_token=4,
                 net_dropout=0,
                 accumulation_steps=1,
                 attention_activation_type="SoftMax",
                 **kwargs):
        super(Zenith, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       **kwargs)
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_token = num_token
        self.accumulation_steps = accumulation_steps
        self.id_dim = id_dim

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
                if feat in ["item_id"]:
                    self.item_info_dim += id_dim
                else:
                    self.item_info_dim += emb_dim
            else:
                if feat in ["user_id"]:
                    continue
                self.non_item_dim += emb_dim

        self.user_id_embedding_layer = FeatureEmbedding(feature_map, id_dim, required_feature_columns=['user_id'])
        self.item_id_embedding_layer = FeatureEmbedding(feature_map, id_dim, required_feature_columns=['item_id'])

        self.feature_embedding_layer = FeatureEmbedding(
            feature_map, embedding_dim,
            not_required_feature_columns=['user_id', 'item_id']
        )

        user_context_dim = self.non_item_dim
        item_attribute_dim = self.item_info_dim * 2
        self.unified_tokenizer_layer = ZenithTokenizer(
            id_dim,
            user_context_dim,
            item_attribute_dim,
            token_dim,
            num_token
        )

        self.attention_layers = MultiHeadTargetAttention(
            input_dim=self.item_info_dim,
            attention_dim=token_dim if attention_dim is None else attention_dim,
            dropout_rate=attention_dropout,
            attention_activation_type=attention_activation_type,
        )

        # Final number of tokens = user_id(1) + item_id(1) + user_context_tokens(num_ns) + item_attribute_tokens(num_ns)
        self.num_prime_tokens = 2 + 2 * num_token

        self.unified_interaction_layers = ZenithBlock(input_dim=token_dim,
                                         num_token=self.num_prime_tokens,
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
        item_attri_emb = self.feature_embedding_layer(item_dict, flatten_emb=True)
        item_attri_emb = item_attri_emb.view(batch_size, -1, self.item_info_dim - self.id_dim)
        item_id_emb = self.item_id_embedding_layer(item_dict, flatten_emb=True)
        item_id_emb = item_id_emb.view(batch_size, -1, self.id_dim)

        item_seq_emb = torch.cat([item_attri_emb, item_id_emb], dim=-1)
        target_emb = item_seq_emb[:, -1, :]      # B x item_info_dim
        sequence_emb = item_seq_emb[:, 0:-1, :]  # B x T x item_info_dim

        # Other non-sequential features -> NS tokens
        user_context_emb = self.feature_embedding_layer(batch_dict, flatten_emb=True)
        user_id_emb = self.user_id_embedding_layer(batch_dict, flatten_emb=True)
        item_attribute = self.attention_layers(target_emb, sequence_emb, mask)
        item_attribute = torch.cat([target_emb, item_attribute], dim=-1)
        unified_tokens = self.unified_tokenizer_layer(user_id_emb.unsqueeze(1), item_id_emb[:, -1, :].unsqueeze(1), user_context_emb, item_attribute)

        # unified model
        unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            unified_tokens
        )

        bottom_output = unified_tokens.mean(dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict


class ZenithTokenizer(nn.Module):
    def __init__(self, id_dim, user_context_dim, item_attribute_dim, token_dim, num_token):
        super(ZenithTokenizer, self).__init__()
        self.user_context_mlps = ChunkTokenizer(user_context_dim, token_dim, num_token)
        self.item_attribute_mlps = ChunkTokenizer(item_attribute_dim, token_dim, num_token)
        self.user_id_mlp = MLP_Block(
            input_dim=id_dim,
            hidden_units=[token_dim],
            hidden_activations="SiLU",
            layer_norm=True,
        )
        self.item_id_mlp = MLP_Block(
            input_dim=id_dim,
            hidden_units=[token_dim],
            hidden_activations="SiLU",
            layer_norm=True,
        )

    def forward(self, user_id_emb, item_id_emb, user_context_emb, item_attribute):
        user_context_tokens = self.user_context_mlps(user_context_emb)
        item_attribute_tokens = self.item_attribute_mlps(item_attribute)
        user_id_token = self.user_id_mlp(user_id_emb)
        item_id_token = self.item_id_mlp(item_id_emb)
        return torch.cat([
            user_id_token,
            item_id_token,
            user_context_tokens,
            item_attribute_tokens,
        ], dim=1)


class ZenithBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_token,
                 num_layers,
                 num_head=2,
                 expand=2,
                 net_dropout=0.0):
        super(ZenithBlock, self).__init__()
        self.num_layers = num_layers

        self.mixer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim)
            for _ in range(num_layers)
        ])
        self.pffn_norms = nn.ModuleList([
            nn.LayerNorm(input_dim)
            for _ in range(num_layers)
        ])

        self.mixer_layers = nn.ModuleList([
            TokenwiseMultiHeadSelfAttention(token_dim=input_dim,
                                            num_token=num_token,
                                            num_head=num_head)
            for _ in range(num_layers)
        ])
        self.pffn_layers = nn.ModuleList([
            PerTokenSwiGLU(input_dim=input_dim,
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

class TokenwiseMultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 token_dim,
                 num_token,
                 num_head=2):
        super(TokenwiseMultiHeadSelfAttention, self).__init__()
        assert token_dim % num_head == 0, "token_dim must be divisible by num_head"
        self.num_head = num_head
        self.head_dim = token_dim // num_head
        self.token_dim = token_dim
        self.num_token = num_token
        self.W_q = nn.Parameter(torch.empty(num_token, token_dim, token_dim))
        self.W_k = nn.Parameter(torch.empty(num_token, token_dim, token_dim))
        self.W_v = nn.Parameter(torch.empty(num_token, token_dim, token_dim))
        self.W_o = nn.Linear(token_dim, token_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, tokens): # B × T × D
        Q = torch.einsum("btd,tdh->bth", tokens, self.W_q)
        K = torch.einsum("btd,tdh->bth", tokens, self.W_k)
        V = torch.einsum("btd,tdh->bth", tokens, self.W_v)

        Q = Q.view(-1, self.num_token, self.num_head, self.head_dim).transpose(1, 2) # [B, H, T, Dh]
        K = K.view(-1, self.num_token, self.num_head, self.head_dim).transpose(1, 2)
        V = V.view(-1, self.num_token, self.num_head, self.head_dim).transpose(1, 2)
        attn_out, _ = self.dot_attention(Q, K, V, scale=self.head_dim ** 0.5)
        attn_out = attn_out.transpose(1, 2).contiguous().view(-1, self.num_token, self.token_dim)  # [B, T, D]
        attn_out = self.W_o(attn_out)
        return attn_out
