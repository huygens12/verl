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
from megatron.core import ModelParallelConfig
from torch import nn
from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn.ops.gemm import FusedDense

from .parallel_linear import ColumnParallelLinear, RowParallelLinear


def get_default_kwargs_for_parallel_linear() -> dict:
    """Get default kwargs for parallel linear layers."""
    return {
        "config": ModelParallelConfig(
            tensor_model_parallel_size=1,
            sequence_parallel_enabled=False,
            use_distributed_optimizer=False,
        )
    }


class PanguMLP(nn.Module):
    """
    MLP layer for Pangu model with parallel linear layers.
    """

    def __init__(self, config, megatron_config: Optional[ModelParallelConfig] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Handle megatron config
        if megatron_config is None:
            from .parallel_linear import get_default_kwargs_for_parallel_linear
            linear_kwargs = get_default_kwargs_for_parallel_linear()
        else:
            linear_kwargs = {"config": megatron_config}

        # Input projection: hidden_size -> intermediate_size * 2 (for gate and up)
        self.gate_up_proj = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size * 2,
            bias=False,
            gather_output=False,
            **linear_kwargs
        )

        # Output projection: intermediate_size -> hidden_size
        self.down_proj = RowParallelLinear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            input_is_parallel=True,
            **linear_kwargs
        )

        # Activation function (SiLU/Swish)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        # Gate and Up projection
        gate_up_output = self.gate_up_proj(hidden_states)

        # Split gate and up outputs
        gate, up = gate_up_output.chunk(2, dim=-1)

        # Apply activation to gate and multiply with up
        intermediate_states = self.act_fn(gate) * up

        # Down projection
        output = self.down_proj(intermediate_states)

        return output


class PanguMLPRmPad(nn.Module):
    """
    MLP layer with remove padding support for Pangu model.
    """

    def __init__(self, config, megatron_config: Optional[ModelParallelConfig] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Handle megatron config
        if megatron_config is None:
            from .parallel_linear import get_default_kwargs_for_parallel_linear
            linear_kwargs = get_default_kwargs_for_parallel_linear()
        else:
            linear_kwargs = {"config": megatron_config}

        # Input projection: hidden_size -> intermediate_size * 2 (for gate and up)
        self.gate_up_proj = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size * 2,
            bias=False,
            gather_output=False,
            **linear_kwargs
        )

        # Output projection: intermediate_size -> hidden_size
        self.down_proj = RowParallelLinear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            input_is_parallel=True,
            **linear_kwargs
        )

        # Activation function (SiLU/Swish)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, attention_mask=None):
        # Gate and Up projection
        gate_up_output = self.gate_up_proj(hidden_states)

        # Split gate and up outputs
        gate, up = gate_up_output.chunk(2, dim=-1)

        # Apply activation to gate and multiply with up
        intermediate_states = self.act_fn(gate) * up

        # Down projection
        output = self.down_proj(intermediate_states)

        return output