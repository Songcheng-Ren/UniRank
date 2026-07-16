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
from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, ScaledDotProductAttention
from unirank.pytorch.torch_utils import get_activation
from unirank.pytorch.layers.tokenization import AutoSplitTokenizer


class LONGER(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="LONGER",
                 task=["binary_classification"],
                 gpu=-1,
                 dnn_activations="ReLU",
                 tower_activations="ReLU",
                 tower_hidden_units=[128, 64],
                 embedding_dim=10,
                 num_layers=3,
                 expansion_factor=4,
                 num_heads=1,
                 num_tasks=4,
                 token_dim=64,
                 num_ns_token=4,
                 num_groups=4,
                 query_size=20,
                 net_dropout=0,
                 accumulation_steps=1,
                 max_len=100,
                 **kwargs):
        super(LONGER, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     **kwargs)


        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.num_ns_token = num_ns_token
        self.accumulation_steps = accumulation_steps
        self.max_len = max_len
        self.num_groups = num_groups
        self.query_size = query_size

        if max_len % num_groups != 0:
            raise ValueError(
                f"[LONGER] sequence length Ls={max_len} is not divisible by num_groups={num_groups}. "
                f"Please truncate/pad in data pipeline or set compatible num_groups."
            )

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

        self.unified_tokenizer_layer = AutoSplitTokenizer(
            input_dim=self.non_item_dim + self.item_info_dim,
            token_dim=token_dim,
            num_tokens=num_ns_token
        )

        if self.item_info_dim != token_dim:
            self.item_token_proj = nn.Linear(self.item_info_dim, token_dim)
        else:
            self.item_token_proj = nn.Identity()

        self.unified_interaction_layers = LONGERBlock(
            input_dim=token_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            num_layers=num_layers,
            query_size=self.query_size,
            expansion_factor=expansion_factor,
            dnn_activations=dnn_activations
        )

        self.tower = nn.ModuleList([
            MLP_Block(input_dim=token_dim * (num_ns_token + self.query_size),
                      output_dim=1,
                      hidden_units=tower_hidden_units,
                      hidden_activations=tower_activations,
                      output_activation=None,
                      dropout_rates=net_dropout)
            for _ in range(num_tasks)
        ])

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

        target_emb = item_seq_emb[:, -1, :]
        sequence_emb = item_seq_emb[:, 0:-1, :]

        s_tokens = self.item_token_proj(sequence_emb)

        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        feature_embeddings = torch.cat([user_context_emb, target_emb], dim=-1)
        unified_tokens = self.unified_tokenizer_layer(feature_embeddings)

        unified_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            s_tokens,
            unified_tokens,
            mask
        )

        bottom_output = unified_tokens.flatten(start_dim=1)
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict


class LONGERBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_heads,
                 num_groups,
                 num_layers,
                 query_size,
                 dnn_activations='ReLU',
                 expansion_factor=4):
        super(LONGERBlock, self).__init__()
        self.num_layers = num_layers
        self.query_size = query_size

        self.inner_trans_norm = nn.LayerNorm(input_dim)
        self.cross_attention_norm = nn.LayerNorm(input_dim)
        self.self_attention_norm = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])

        self.inner_trans = InnerTrans(
            input_dim=input_dim,
            num_groups=num_groups,
            expansion_factor=expansion_factor,
            dnn_activations=dnn_activations
        )

        self.cross_causal_attention = CrossCausalAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            dnn_activations=dnn_activations
        )

        self.self_attention_layers = nn.ModuleList([
            SelfCausalAttention(
                input_dim=input_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                dnn_activations=dnn_activations
            ) for _ in range(num_layers)
        ])

    def forward(self, s_tokens, ns_tokens, mask=None):
        """
        s_tokens: B x Ls x D
        ns_tokens: B x Lns x D
        mask: B x Ls
        """
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).float()

        merge_s_tokens = self.inner_trans(self.inner_trans_norm(s_tokens))  # B x Lms x D

        sampled_s_tokens = s_tokens[:, -self.query_size:, :]

        q_tokens = torch.cat([sampled_s_tokens, ns_tokens], dim=1)
        kv_tokens = torch.cat([merge_s_tokens, ns_tokens], dim=1)

        tokens = self.cross_causal_attention(
            self.cross_attention_norm(q_tokens),
            self.cross_attention_norm(kv_tokens)
        ) + q_tokens

        for i in range(self.num_layers):
            tokens = self.self_attention_layers[i](self.self_attention_norm[i](tokens)) + tokens
        return tokens


class InnerTrans(nn.Module):
    def __init__(self, input_dim, num_groups=4, expansion_factor=4, dnn_activations='ReLU'):
        super(InnerTrans, self).__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        hidden_dim = input_dim * num_groups
        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention()
        self.FFN = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            get_activation(dnn_activations),
            nn.Linear(hidden_dim * expansion_factor, input_dim)
        )

    def forward(self, s_tokens):
        """
        s_tokens: B x Ls x D
        return:
            merge_s_tokens: B x Lms x D
        """
        B, Ls, D = s_tokens.shape

        if Ls % self.num_groups != 0:
            raise ValueError(
                f"[InnerTrans] sequence length Ls={Ls} is not divisible by num_groups={self.num_groups}."
            )

        Lms = Ls // self.num_groups

        # Group adjacent tokens in chunks of num_groups
        merge_s_tokens = s_tokens.reshape(B, Lms, self.num_groups, D)

        merge_s_tokens = merge_s_tokens.reshape(B * Lms, self.num_groups, D)

        Q = self.W_q(merge_s_tokens)
        K = self.W_k(merge_s_tokens)
        V = self.W_v(merge_s_tokens)

        output, _ = self.dot_attention(Q, K, V, scale=self.input_dim ** 0.5)

        output = output.reshape(B, Lms, self.num_groups * D)
        merge_s_tokens = self.FFN(output)

        return merge_s_tokens


class CrossCausalAttention(nn.Module):
    def __init__(self, input_dim, num_heads, expansion_factor=4, dnn_activations='ReLU'):
        super(CrossCausalAttention, self).__init__()
        assert input_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(input_dim, num_heads)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention()
        self.FFN = nn.Sequential(
            nn.Linear(input_dim, input_dim * expansion_factor),
            get_activation(dnn_activations),
            nn.Linear(input_dim * expansion_factor, input_dim)
        )

    def forward(self, q_tokens, kv_tokens):
        """
        q_tokens: B x (Lq + Lns) x D
        kv_tokens: B x (Lms + Lns) x D
        """
        B, _, D = q_tokens.shape

        Q = self.W_q(q_tokens)
        K = self.W_k(kv_tokens)
        V = self.W_v(kv_tokens)

        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        output, _ = self.dot_attention(Q, K, V, scale=self.head_dim ** 0.5)

        output = output.transpose(1, 2).contiguous().view(B, -1, D)
        output = self.FFN(output)
        return output


class SelfCausalAttention(nn.Module):
    def __init__(self, input_dim, num_heads, expansion_factor=4, dnn_activations='ReLU'):
        super(SelfCausalAttention, self).__init__()
        assert input_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(input_dim, num_heads)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention()
        self.FFN = nn.Sequential(
            nn.Linear(input_dim, input_dim * expansion_factor),
            get_activation(dnn_activations),
            nn.Linear(input_dim * expansion_factor, input_dim)
        )

    def forward(self, tokens):
        """
        tokens: B x (Lq + Lns) x D
        """
        B, L, D = tokens.shape

        Q = self.W_q(tokens)
        K = self.W_k(tokens)
        V = self.W_v(tokens)

        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        output, _ = self.dot_attention(
            Q,
            K,
            V,
            scale=self.head_dim ** 0.5,
            is_causal=True,
        )

        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.FFN(output)
        return output
