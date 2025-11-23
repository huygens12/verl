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

from typing import Optional, Tuple

import torch
from megatron.core import ModelParallelConfig, tensor_parallel
from megatron.core.tensor_parallel.utils import divide
from torch import nn


def get_default_kwargs_for_parallel_embedding() -> dict:
    """Get default kwargs for parallel embedding."""
    return {
        "config": ModelParallelConfig(
            tensor_model_parallel_size=1,
            sequence_parallel_enabled=False,
            use_distributed_optimizer=False,
        )
    }


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Optional[callable] = None,
        bias_init: Optional[callable] = None,
        config: Optional[ModelParallelConfig] = None,
        skip_weight_param_allocation: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        # Model parallel config
        if config is None:
            config = ModelParallelConfig()
        self.config = config

        # Divide output dimension
        world_size = tensor_parallel.get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(out_features, world_size)

        # Skip weight allocation if specified
        if not skip_weight_param_allocation:
            self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, in_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            else:
                self.register_parameter("bias", None)

        # Initialize weights
        if init_method is not None and not skip_weight_param_allocation:
            init_method(self.weight)
        if bias_init is not None and bias is not None:
            bias_init(self.bias)

        # Tensor parallel settings
        self.tensor_parallel_config = config

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Set up bias
        bias = self.bias if not self.gather_output else self.bias

        # Linear forward pass
        output = tensor_parallel.linear_with_grad_bias(
            input_, self.weight, bias, self.tensor_parallel_config.tensor_model_parallel_rank
        )

        if self.gather_output:
            # All-reduce across all the columns
            output = tensor_parallel.all_reduce(output, self.tensor_parallel_config.tensor_model_parallel_group)

        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_is_parallel: bool = True,
        bias: bool = True,
        init_method: Optional[callable] = None,
        bias_init: Optional[callable] = None,
        config: Optional[ModelParallelConfig] = None,
        skip_weight_param_allocation: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        # Model parallel config
        if config is None:
            config = ModelParallelConfig()
        self.config = config

        # Divide input dimension
        world_size = tensor_parallel.get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(in_features, world_size)

        # Skip weight allocation if specified
        if not skip_weight_param_allocation:
            self.weight = nn.Parameter(torch.empty(self.out_features, self.input_size_per_partition))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.out_features))
            else:
                self.register_parameter("bias", None)

        # Initialize weights
        if init_method is not None and not skip_weight_param_allocation:
            init_method(self.weight.T)
        if bias_init is not None and bias is not None:
            bias_init(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Linear forward pass
        output_parallel = tensor_parallel.linear_with_grad_bias(
            input_, self.weight.T, self.bias, tensor_parallel.get_tensor_model_parallel_rank()
        )

        # All-reduce if input was parallel
        if self.input_is_parallel:
            output = tensor_parallel.all_reduce(output_parallel, self.config.tensor_model_parallel_group)
        else:
            output = output_parallel

        return output


class QKVParallelLinear(nn.Module):
    """
    Combined QKV linear layer for attention.
    """

    def __init__(
        self,
        hidden_size: int,
        total_num_heads: int,
        head_size: int,
        bias: bool = True,
        config: Optional[ModelParallelConfig] = None,
        skip_weight_param_allocation: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.head_size = head_size

        # Model parallel config
        if config is None:
            config = ModelParallelConfig()
        self.config = config

        # Calculate dimensions
        world_size = tensor_parallel.get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(total_num_heads, world_size)
        self.hidden_size_per_partition = hidden_size
        self.output_size_per_partition = self.num_heads_per_partition * head_size * 3  # Q, K, V

        # Skip weight allocation if specified
        if not skip_weight_param_allocation:
            self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.hidden_size_per_partition))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            else:
                self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Linear forward pass
        output_parallel = tensor_parallel.linear_with_grad_bias(
            hidden_states,
            self.weight,
            self.bias,
            self.config.tensor_model_parallel_rank
        )

        # All-reduce across all the columns
        output = tensor_parallel.all_reduce(output_parallel, self.config.tensor_model_parallel_group)

        # Reshape for Q, K, V
        new_shape = output.size()[:-1] + (self.num_heads_per_partition, 3, self.head_size)
        output = output.view(*new_shape)

        # Split Q, K, V
        query_states = output[..., 0, :]
        key_states = output[..., 1, :]
        value_states = output[..., 2, :]

        return query_states, key_states, value_states