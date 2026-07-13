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

class MultiHeadTokenMixing(nn.Module):
    def __init__(self, input_dim, num_token):
        super(MultiHeadTokenMixing, self).__init__()
        self.num_token = num_token # = num_heads
        self.input_dim = input_dim
        assert input_dim % num_token == 0, "input_dim must be divisible by num_tokens"
        self.head_dim = self.input_dim // self.num_token

    def forward(self, x):  # x: [B, T, D]
        heads = torch.tensor_split(x, self.num_token, dim=-1)  # list(H) of [B, T, Dh]
        mixed = torch.stack(heads, dim=1)                      # [B, H, T, Dh]
        out = mixed.flatten(start_dim=2)                     # [B, H, T*Dh]，当 H=T, T*Dh=D -> [B,T,D]
        return out
