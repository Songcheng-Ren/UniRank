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
import torch.nn.functional as F
from torch import nn

from unirank.pytorch.layers import FeatureEmbedding, MLP_Block, ScaledDotProductAttention
from unirank.pytorch.layers.tokenization import build_unified_tokenizer
from unirank.pytorch.models import MultiTaskModel

from .OneTrans import MixedFFN


class OneTrans_var(MultiTaskModel):
    """OneTrans with independently configurable architectural enhancements."""

    def __init__(self,
                 feature_map,
                 model_id="OneTrans_var",
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
                 tokenizer_type="Auto",
                 attention_activation_type="SoftMax",
                 pre_u=False,
                 post_u=False,
                 fuse_u=True,
                 spec_t=False,
                 att_gate=False,
                 rope=False,
                 qk_norm=False,
                 g_norm=False,
                 net_dropout=0,
                 accumulation_steps=1,
                 **kwargs):
        super(OneTrans_var, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            **kwargs
        )
        self.num_tasks = num_tasks
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.accumulation_steps = accumulation_steps

        self.pre_u = bool(pre_u)
        self.post_u = bool(post_u)
        self.fuse_u = bool(fuse_u)
        self.spec_t = bool(spec_t)
        self.att_gate = bool(att_gate)
        self.rope = bool(rope)
        self.qk_norm = bool(qk_norm)
        self.g_norm = bool(g_norm)

        user_placement_count = sum((self.pre_u, self.post_u, self.fuse_u))
        if user_placement_count > 1:
            raise ValueError(
                "pre_u, post_u, and fuse_u are mutually exclusive user-placement strategies."
            )

        self.item_info_dim = 0
        self.non_item_dim = 0
        self.num_item_fields = 0
        self.num_non_item_fields = 0
        for feat, spec in self.feature_map.features.items():
            if feat in self.feature_map.labels or spec.get("type") == "meta":
                continue
            emb_dim = spec.get("embedding_dim", embedding_dim)
            if spec.get("source") in ["item", "action"]:
                self.item_info_dim += emb_dim
                self.num_item_fields += 1
            else:
                self.non_item_dim += emb_dim
                self.num_non_item_fields += 1

        if (self.pre_u or self.post_u) and self.non_item_dim == 0:
            raise ValueError("PreU and PostU require at least one user/context feature.")

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        tokenizer_input_dim = self.item_info_dim
        self.num_tokenizer_fields = self.num_item_fields
        if self.fuse_u:
            tokenizer_input_dim += self.non_item_dim
            self.num_tokenizer_fields += self.num_non_item_fields

        (self.unified_tokenizer_layer,
         self.num_ns_token,
         self.tokenizer_uses_field_input) = build_unified_tokenizer(
            tokenizer_type=tokenizer_type,
            input_dim=tokenizer_input_dim,
            field_dim=embedding_dim,
            token_dim=token_dim,
            num_tokens=num_ns_token,
            num_fields=self.num_tokenizer_fields,
            layer_norm=not self.g_norm,
        )
        self.tokenizer_type = str(tokenizer_type).strip().title()

        self.item_token_proj = (
            nn.Linear(self.item_info_dim, token_dim)
            if self.item_info_dim != token_dim
            else nn.Identity()
        )

        if self.pre_u or self.post_u:
            self.user_token_proj = nn.Sequential(
                nn.Linear(self.non_item_dim, token_dim),
                nn.SiLU(),
            )
        else:
            self.user_token_proj = None

        if self.spec_t:
            self.bos_token = nn.Parameter(torch.empty(1, 1, token_dim))
            self.sep_token = nn.Parameter(torch.empty(1, 1, token_dim))
            nn.init.xavier_uniform_(self.bos_token)
            nn.init.xavier_uniform_(self.sep_token)
        else:
            self.register_parameter("bos_token", None)
            self.register_parameter("sep_token", None)

        self.num_prefix_non_sequence_tokens = (
            int(self.pre_u) + int(self.post_u) + 2 * int(self.spec_t)
        )
        self.num_pre_sequence_tokens = (
            int(self.pre_u) + int(self.spec_t)
        )
        self.num_post_sequence_tokens = (
            int(self.post_u) + int(self.spec_t)
        )

        if self.g_norm:
            self.sequence_tokenizer_norm = nn.LayerNorm(token_dim)
            self.ns_tokenizer_norm = PerTokenLayerNorm(self.num_ns_token, token_dim)
            self.user_tokenizer_norm = (
                nn.LayerNorm(token_dim)
                if self.pre_u or self.post_u
                else nn.Identity()
            )
        else:
            self.sequence_tokenizer_norm = nn.Identity()
            self.ns_tokenizer_norm = nn.Identity()
            self.user_tokenizer_norm = nn.Identity()

        self.unified_interaction_layers = OneTransVarBlock(
            input_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_ns_token=self.num_ns_token,
            num_prefix_non_sequence_tokens=self.num_prefix_non_sequence_tokens,
            num_pre_sequence_tokens=self.num_pre_sequence_tokens,
            num_post_sequence_tokens=self.num_post_sequence_tokens,
            dnn_activations=dnn_activations,
            expansion_factor=expansion_factor,
            attention_activation_type=attention_activation_type,
            att_gate=self.att_gate,
            rope=self.rope,
            qk_norm=self.qk_norm,
            g_norm=self.g_norm,
        )

        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=token_dim * self.num_ns_token,
                output_dim=1,
                hidden_units=tower_hidden_units,
                hidden_activations=tower_activations,
                output_activation=None,
                dropout_rates=net_dropout,
            )
            for _ in range(num_tasks)
        ])
        if isinstance(task, list):
            assert len(task) == num_tasks, \
                "the number of tasks must equal the length of \"task\""
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
        self.model_to_device()

    def _build_sequence_stream(self, sequence_tokens, user_token, seq_mask):
        batch_size, seq_len, _ = sequence_tokens.shape
        device = sequence_tokens.device
        dtype = sequence_tokens.dtype

        token_parts = []
        mask_parts = []
        type_id_parts = []
        position_parts = []
        next_non_sequence_id = 0

        def append_non_sequence(token):
            nonlocal next_non_sequence_id
            token_parts.append(token)
            mask_parts.append(torch.ones(
                batch_size, token.size(1), dtype=torch.bool, device=device
            ))
            type_id_parts.append(torch.full(
                (token.size(1),),
                next_non_sequence_id,
                dtype=torch.long,
                device=device,
            ))
            position_parts.append(torch.full(
                (batch_size, token.size(1)),
                -1,
                dtype=torch.long,
                device=device,
            ))
            next_non_sequence_id += token.size(1)

        if self.pre_u:
            append_non_sequence(user_token)
        if self.spec_t:
            append_non_sequence(self.bos_token.to(dtype=dtype).expand(batch_size, -1, -1))

        seq_mask = seq_mask.bool()
        sequence_positions = seq_mask.long().cumsum(dim=1) - 1
        sequence_positions = sequence_positions.masked_fill(~seq_mask, -1)
        token_parts.append(sequence_tokens)
        mask_parts.append(seq_mask)
        type_id_parts.append(torch.full(
            (seq_len,), -1, dtype=torch.long, device=device
        ))
        position_parts.append(sequence_positions)

        if self.post_u:
            append_non_sequence(user_token)
        if self.spec_t:
            append_non_sequence(self.sep_token.to(dtype=dtype).expand(batch_size, -1, -1))

        return (
            torch.cat(token_parts, dim=1),
            torch.cat(mask_parts, dim=1),
            torch.cat(type_id_parts, dim=0),
            torch.cat(position_parts, dim=1),
        )

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        batch_size = mask.shape[0]

        item_seq_emb = self.embedding_layer(item_dict, flatten_emb=True)
        item_seq_emb = item_seq_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_seq_emb[:, -1, :]
        sequence_emb = item_seq_emb[:, :-1, :]

        sequence_tokens = self.sequence_tokenizer_norm(
            self.item_token_proj(sequence_emb)
        )

        user_context_emb = self.embedding_layer(batch_dict, flatten_emb=True)
        user_token = None
        if self.pre_u or self.post_u:
            user_token = self.user_tokenizer_norm(
                self.user_token_proj(user_context_emb)
            ).unsqueeze(1)

        tokenizer_input = target_emb
        if self.fuse_u:
            tokenizer_input = torch.cat([user_context_emb, target_emb], dim=-1)
        if self.tokenizer_uses_field_input:
            tokenizer_input = tokenizer_input.reshape(
                batch_size,
                self.num_tokenizer_fields,
                self.embedding_dim,
            )
        ns_tokens = self.ns_tokenizer_norm(
            self.unified_tokenizer_layer(tokenizer_input)
        )

        (sequence_stream,
         sequence_mask,
         sequence_type_ids,
         sequence_positions) = self._build_sequence_stream(
            sequence_tokens,
            user_token,
            mask,
        )

        ns_tokens = self.activation_checkpoint(
            self.unified_interaction_layers,
            sequence_stream,
            ns_tokens,
            sequence_mask,
            sequence_type_ids,
            sequence_positions,
        )

        bottom_output = ns_tokens.flatten(start_dim=1)
        tower_output = [
            self.tower[index](bottom_output)
            for index in range(self.num_tasks)
        ]
        y_pred = [
            self.output_activation[index](tower_output[index])
            for index in range(self.num_tasks)
        ]
        labels = self.feature_map.labels
        return {
            "{}_pred".format(labels[index]): y_pred[index]
            for index in range(self.num_tasks)
        }


class PerTokenLayerNorm(nn.Module):
    """LayerNorm with independent affine parameters for each token."""

    def __init__(self, num_tokens, input_dim, eps=1e-5):
        super(PerTokenLayerNorm, self).__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_tokens, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_tokens, input_dim))

    def forward(self, tokens):
        if tokens.size(1) != self.weight.size(0):
            raise ValueError(
                "Expected {} tokens, got {}.".format(
                    self.weight.size(0), tokens.size(1)
                )
            )
        normalized = F.layer_norm(
            tokens,
            (self.input_dim,),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        return normalized * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class SequenceAwareLayerNorm(nn.Module):
    """Shared sequence norm plus independent norms for non-sequence tokens."""

    def __init__(self,
                 input_dim,
                 num_prefix_non_sequence_tokens,
                 num_ns_token,
                 enabled=False,
                 eps=1e-5):
        super(SequenceAwareLayerNorm, self).__init__()
        self.enabled = enabled
        self.input_dim = input_dim
        self.num_prefix_non_sequence_tokens = num_prefix_non_sequence_tokens
        self.num_ns_token = num_ns_token
        self.eps = eps

        if enabled:
            num_non_sequence_tokens = (
                num_prefix_non_sequence_tokens + num_ns_token
            )
            self.sequence_weight = nn.Parameter(torch.ones(input_dim))
            self.sequence_bias = nn.Parameter(torch.zeros(input_dim))
            self.non_sequence_weight = nn.Parameter(torch.ones(
                num_non_sequence_tokens, input_dim
            ))
            self.non_sequence_bias = nn.Parameter(torch.zeros(
                num_non_sequence_tokens, input_dim
            ))
        else:
            self.shared_norm = nn.LayerNorm(input_dim)

    def _normalize(self, tokens):
        return F.layer_norm(
            tokens,
            (self.input_dim,),
            weight=None,
            bias=None,
            eps=self.eps,
        )

    def normalize_sequence_stream(self, tokens, token_type_ids):
        if not self.enabled:
            return self.shared_norm(tokens)

        normalized = self._normalize(tokens)
        safe_type_ids = token_type_ids.clamp_min(0)
        non_sequence_weight = self.non_sequence_weight.index_select(
            0, safe_type_ids
        )
        non_sequence_bias = self.non_sequence_bias.index_select(
            0, safe_type_ids
        )
        is_non_sequence = token_type_ids.ge(0).view(1, -1, 1)
        weight = torch.where(
            is_non_sequence,
            non_sequence_weight.unsqueeze(0),
            self.sequence_weight.view(1, 1, -1),
        )
        bias = torch.where(
            is_non_sequence,
            non_sequence_bias.unsqueeze(0),
            self.sequence_bias.view(1, 1, -1),
        )
        return normalized * weight + bias

    def normalize_ns_tokens(self, tokens):
        if not self.enabled:
            return self.shared_norm(tokens)

        normalized = self._normalize(tokens)
        start = self.num_prefix_non_sequence_tokens
        stop = start + self.num_ns_token
        weight = self.non_sequence_weight[start:stop].unsqueeze(0)
        bias = self.non_sequence_bias[start:stop].unsqueeze(0)
        return normalized * weight + bias


class RotaryEmbedding(nn.Module):
    """RoPE applied only where chronological position ids are non-negative."""

    def __init__(self, head_dim, base=10000.0):
        super(RotaryEmbedding, self).__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, tensor, position_ids):
        valid_positions = position_ids.ge(0)
        safe_positions = position_ids.clamp_min(0).to(self.inv_freq.dtype)
        angles = (
            safe_positions.unsqueeze(1).unsqueeze(-1)
            * self.inv_freq.view(1, 1, 1, -1)
        )
        cos = angles.cos().to(dtype=tensor.dtype)
        sin = angles.sin().to(dtype=tensor.dtype)

        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        ).flatten(start_dim=-2)
        return torch.where(
            valid_positions.unsqueeze(1).unsqueeze(-1),
            rotated,
            tensor,
        )


class OneTransVarBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_heads,
                 num_layers,
                 num_ns_token,
                 num_prefix_non_sequence_tokens,
                 num_pre_sequence_tokens,
                 num_post_sequence_tokens,
                 dnn_activations="ReLU",
                 expansion_factor=4,
                 attention_activation_type="SoftMax",
                 att_gate=False,
                 rope=False,
                 qk_norm=False,
                 g_norm=False):
        super(OneTransVarBlock, self).__init__()
        self.num_layers = num_layers
        self.num_pre_sequence_tokens = num_pre_sequence_tokens
        self.num_post_sequence_tokens = num_post_sequence_tokens

        norm_kwargs = dict(
            input_dim=input_dim,
            num_prefix_non_sequence_tokens=num_prefix_non_sequence_tokens,
            num_ns_token=num_ns_token,
            enabled=g_norm,
        )
        self.attention_norms = nn.ModuleList([
            SequenceAwareLayerNorm(**norm_kwargs)
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            SequenceAwareLayerNorm(**norm_kwargs)
            for _ in range(num_layers)
        ])
        self.mha_layers = nn.ModuleList([
            VarMixedMHA(
                input_dim=input_dim,
                num_heads=num_heads,
                num_ns_token=num_ns_token,
                num_pre_sequence_tokens=num_pre_sequence_tokens,
                attention_activation_type=attention_activation_type,
                att_gate=att_gate,
                rope=rope,
                qk_norm=qk_norm,
            )
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            MixedFFN(
                input_dim=input_dim,
                num_ns_token=num_ns_token,
                dnn_activations=dnn_activations,
                expansion_factor=expansion_factor,
            )
            for _ in range(num_layers)
        ])

    def forward(self,
                s_tokens,
                ns_tokens,
                mask,
                token_type_ids,
                sequence_positions):
        mask = mask.bool()
        s_tokens = s_tokens * mask.unsqueeze(-1).to(dtype=s_tokens.dtype)

        ps_tokens = s_tokens
        q_mask = mask
        ps_token_type_ids = token_type_ids
        ps_sequence_positions = sequence_positions

        for index in range(self.num_layers):
            num_sequence_tokens = (
                ps_tokens.size(1)
                - self.num_pre_sequence_tokens
                - self.num_post_sequence_tokens
            )
            sequence_start = (
                self.num_pre_sequence_tokens + num_sequence_tokens // 2
            )
            ps_tokens = torch.cat([
                ps_tokens[:, :self.num_pre_sequence_tokens, :],
                ps_tokens[:, sequence_start:, :],
            ], dim=1)
            q_mask = torch.cat([
                q_mask[:, :self.num_pre_sequence_tokens],
                q_mask[:, sequence_start:],
            ], dim=1)
            ps_token_type_ids = torch.cat([
                ps_token_type_ids[:self.num_pre_sequence_tokens],
                ps_token_type_ids[sequence_start:],
            ], dim=0)
            ps_sequence_positions = torch.cat([
                ps_sequence_positions[:, :self.num_pre_sequence_tokens],
                ps_sequence_positions[:, sequence_start:],
            ], dim=1)

            attention_norm = self.attention_norms[index]
            norm_s = attention_norm.normalize_sequence_stream(
                s_tokens, token_type_ids
            )
            norm_ps = attention_norm.normalize_sequence_stream(
                ps_tokens, ps_token_type_ids
            )
            norm_ns = attention_norm.normalize_ns_tokens(ns_tokens)

            delta_ps, delta_ns = self.mha_layers[index](
                norm_s,
                norm_ps,
                norm_ns,
                kv_mask=mask,
                q_mask=q_mask,
                sequence_positions=sequence_positions,
            )
            ps_tokens = ps_tokens + delta_ps
            ns_tokens = ns_tokens + delta_ns

            ffn_norm = self.ffn_norms[index]
            norm_ps = ffn_norm.normalize_sequence_stream(
                ps_tokens, ps_token_type_ids
            )
            norm_ns = ffn_norm.normalize_ns_tokens(ns_tokens)
            delta_ps, delta_ns = self.ffn_layers[index](
                norm_ps,
                norm_ns,
                q_mask,
            )
            ps_tokens = ps_tokens + delta_ps
            ns_tokens = ns_tokens + delta_ns

            s_tokens = ps_tokens
            mask = q_mask
            token_type_ids = ps_token_type_ids
            sequence_positions = ps_sequence_positions

        return ns_tokens


class VarMixedMHA(nn.Module):
    def __init__(self,
                 input_dim,
                 num_heads,
                 num_ns_token,
                 num_pre_sequence_tokens=0,
                 attention_activation_type="SoftMax",
                 att_gate=False,
                 rope=False,
                 qk_norm=False):
        super(VarMixedMHA, self).__init__()
        if input_dim % num_heads != 0:
            raise ValueError(
                "input_dim={} is not divisible by num_heads={}."
                .format(input_dim, num_heads)
            )
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_ns_token = num_ns_token
        self.num_pre_sequence_tokens = num_pre_sequence_tokens
        self.head_dim = input_dim // num_heads
        self.use_att_gate = bool(att_gate)
        self.use_rope = bool(rope)
        self.use_qk_norm = bool(qk_norm)

        self.W_q_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_q_ns = nn.Parameter(torch.empty(
            num_ns_token, input_dim, input_dim
        ))
        self.W_k_ns = nn.Parameter(torch.empty(
            num_ns_token, input_dim, input_dim
        ))
        self.W_v_ns = nn.Parameter(torch.empty(
            num_ns_token, input_dim, input_dim
        ))
        self.W_o = nn.Linear(input_dim, input_dim, bias=False)

        if self.use_att_gate:
            self.W_g = nn.Sequential(
                nn.Linear(input_dim, input_dim, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.W_g = None

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.rotary = (
            RotaryEmbedding(self.head_dim)
            if self.use_rope
            else None
        )
        self.dot_attention = ScaledDotProductAttention(
            attention_activation_type=attention_activation_type
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q_ns)
        nn.init.xavier_uniform_(self.W_k_ns)
        nn.init.xavier_uniform_(self.W_v_ns)

    def forward(self,
                s_tokens,
                ps_tokens,
                ns_tokens,
                kv_mask=None,
                q_mask=None,
                sequence_positions=None):
        batch_size, sequence_length, input_dim = s_tokens.shape
        query_sequence_length = ps_tokens.size(1)
        ns_length = ns_tokens.size(1)

        if kv_mask is not None:
            s_tokens = s_tokens * kv_mask.unsqueeze(-1).to(
                dtype=s_tokens.dtype
            )

        q_s = self.W_q_s(s_tokens)
        k_s = self.W_k_s(s_tokens)
        v_s = self.W_v_s(s_tokens)
        q_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_q_ns)
        k_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_k_ns)
        v_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_v_ns)

        q = torch.cat([q_s, q_ns], dim=1)
        k = torch.cat([k_s, k_ns], dim=1)
        v = torch.cat([v_s, v_ns], dim=1)
        total_length = sequence_length + ns_length

        q = q.view(
            batch_size, total_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, total_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, total_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.rotary is not None:
            ns_positions = torch.full(
                (batch_size, ns_length),
                -1,
                dtype=torch.long,
                device=s_tokens.device,
            )
            full_positions = torch.cat(
                [sequence_positions, ns_positions],
                dim=1,
            )
            q = self.rotary(q, full_positions)
            k = self.rotary(k, full_positions)

        output, _ = self.dot_attention(
            q,
            k,
            v,
            scale=self.head_dim ** 0.5,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, total_length, input_dim
        )

        if self.W_g is not None:
            gate_input = torch.cat([s_tokens, ns_tokens], dim=1)
            output = output * self.W_g(gate_input)
        output = self.W_o(output)

        removed_sequence_length = sequence_length - query_sequence_length
        sequence_query_start = (
            self.num_pre_sequence_tokens + removed_sequence_length
        )
        ps_out = torch.cat([
            output[:, :self.num_pre_sequence_tokens, :],
            output[:, sequence_query_start:sequence_length, :],
        ], dim=1)
        ns_out = output[:, sequence_length:, :]
        if q_mask is not None:
            ps_out = ps_out * q_mask.unsqueeze(-1).to(dtype=ps_out.dtype)
        return ps_out, ns_out
