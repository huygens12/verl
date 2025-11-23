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
from megatron.core import ModelParallelConfig
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from .parallel_attention import PanguAttention, PanguAttentionRmPad
from .parallel_mlp import PanguMLP, PanguMLPRmPad
from .parallel_rmsnorm import PanguRMSNorm, PanguRMSNormRmPad


class PanguDecoderLayer(nn.Module):
    """
    Transformer decoder layer for Pangu model with tensor parallelism.
    """

    def __init__(self, config, megatron_config: Optional[ModelParallelConfig] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PanguAttention(config, megatron_config)
        self.mlp = PanguMLP(config, megatron_config)
        self.input_layernorm = PanguRMSNorm(config, megatron_config)
        self.post_attention_layernorm = PanguRMSNorm(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # Input layernorm
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = outputs[0]
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + outputs[1:]

        return outputs


class PanguDecoderLayerRmPad(nn.Module):
    """
    Transformer decoder layer with remove padding support for Pangu model.
    """

    def __init__(self, config, megatron_config: Optional[ModelParallelConfig] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PanguAttentionRmPad(config, megatron_config)
        self.mlp = PanguMLPRmPad(config, megatron_config)
        self.input_layernorm = PanguRMSNormRmPad(config, megatron_config)
        self.post_attention_layernorm = PanguRMSNormRmPad(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(total_tokens, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask for remove padding
            position_ids (`torch.LongTensor`, *optional*): position indices for each token
        """

        residual = hidden_states

        # Input layernorm
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states