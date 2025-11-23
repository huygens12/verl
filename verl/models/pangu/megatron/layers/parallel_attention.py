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

import math
from typing import Optional

import torch
from megatron.core import ModelParallelConfig, parallel_state, tensor_parallel
from torch import nn

from .parallel_linear import QKVParallelLinear, RowParallelLinear


class PanguRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for Pangu model.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute the rotary frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]

        # Generate frequencies for the given sequence length
        t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Create the cos and sin embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Apply rotary positional embeddings to query and key tensors.
    """
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Apply cos and sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class PanguAttention(nn.Module):
    """
    Multi-head attention layer for Pangu model with tensor parallelism.
    """

    def __init__(self, config, megatron_config: Optional[ModelParallelConfig] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.is_causal = True

        # Tensor parallel setup
        if megatron_config is None:
            megatron_config = ModelParallelConfig()
        self.megatron_config = megatron_config

        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            total_num_heads=self.num_heads,
            head_size=self.head_dim,
            bias=False,
            config=megatron_config
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            input_is_parallel=True,
            config=megatron_config
        )

        # Rotary embeddings
        self.rotary_emb = PanguRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[tuple[torch.Tensor]]]]:
        batch_size, seq_length, _ = hidden_states.size()

        # QKV projection
        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        # Reshape for attention
        query_states = query_states.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_length)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key value for caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_weights = attn_weights + attention_mask

        # Causal mask (for autoregressive generation)
        if self.is_causal and seq_length > 1:
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=torch.bool, device=attn_weights.device),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Softmax attention weights
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and transpose back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_value,)

        return outputs


class PanguAttentionRmPad(nn.Module):
    """
    Multi-head attention layer with remove padding support for Pangu model.
    """

    def __init__(self, config, megatron_config: Optional[ModelParallelConfig] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)

        # Tensor parallel setup
        if megatron_config is None:
            megatron_config = ModelParallelConfig()
        self.megatron_config = megatron_config

        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            total_num_heads=self.num_heads,
            head_size=self.head_dim,
            bias=False,
            config=megatron_config
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            input_is_parallel=True,
            config=megatron_config
        )

        # Rotary embeddings
        self.rotary_emb = PanguRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # For remove padding version, input is already flattened
        total_tokens, _ = hidden_states.size()

        # QKV projection
        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        # Reshape for attention: [total_tokens, num_heads, head_dim]
        query_states = query_states.view(total_tokens, -1, self.head_dim)
        key_states = key_states.view(total_tokens, -1, self.head_dim)
        value_states = value_states.view(total_tokens, -1, self.head_dim)

        # Apply rotary embeddings (need sequence information)
        if position_ids is not None:
            cos, sin = self.rotary_emb(hidden_states, seq_len=position_ids.max() + 1)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos.squeeze(), sin.squeeze(), position_ids
            )

        # For simplicity, this is a basic implementation
        # In practice, you'd need more complex handling for remove padding
        attn_output = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_output = attn_output + attention_mask

        attn_weights = torch.softmax(attn_output, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        # Flatten back to [total_tokens, hidden_size]
        attn_output = attn_output.contiguous().view(total_tokens, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output