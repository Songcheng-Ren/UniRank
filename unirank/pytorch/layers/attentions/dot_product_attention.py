# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) 2018. pengshuang@Github for ScaledDotProductAttention.
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


import math
import torch.nn.functional as F
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention 
        Ref: https://zhuanlan.zhihu.com/p/47812375
    """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, Q, K, V, scale=None, mask=None):
        # mask: 0 for masked positions
        query = Q
        if mask is not None:
            mask = mask.bool()
        if scale is None:
            query = query * math.sqrt(Q.size(-1))
        elif scale != math.sqrt(Q.size(-1)):
            query = query * (math.sqrt(Q.size(-1)) / scale)
        output = F.scaled_dot_product_attention(
            query,
            K,
            V,
            attn_mask=mask,
            dropout_p=self.dropout_rate if self.training else 0.0
        )
        return output, None

