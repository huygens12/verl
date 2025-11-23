# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from typing import Optional

import torch
from megatron.core import ModelParallelConfig, tensor_parallel
from torch import nn


class PanguRMSNorm(nn.Module):
    """
    RMS Normalization for Pangu model.
    """

    def __init__(self, config, megatron_config: Optional[ModelParallelConfig] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.eps = config.rms_norm_eps if hasattr(config, 'rms_norm_eps') else 1e-6
        self.weight = nn.Parameter(torch.ones(self.hidden_size))
        self.megatron_config = megatron_config

    def forward(self, hidden_states):
        # Input tensor shape: [batch_size, seq_len, hidden_size]
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class PanguRMSNormRmPad(nn.Module):
    """
    RMS Normalization with remove padding support for Pangu model.
    """

    def __init__(self, config, megatron_config: Optional[ModelParallelConfig] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.eps = config.rms_norm_eps if hasattr(config, 'rms_norm_eps') else 1e-6
        self.weight = nn.Parameter(torch.ones(self.hidden_size))
        self.megatron_config = megatron_config

    def forward(self, hidden_states, attention_mask=None):
        # Input tensor shape: [total_tokens, hidden_size] after remove padding
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)