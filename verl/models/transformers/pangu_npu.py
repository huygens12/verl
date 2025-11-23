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

"""NPU optimizations for Pangu model."""

import torch
import torch.nn as nn
from verl.utils.device import is_npu_available

if is_npu_available:
    import torch_npu
    from torch_npu import npu_rotary_mul as apply_rotary_emb
    from transformers.integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
    from transformers.integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func


class NPUPanguRMSNorm(nn.Module):
    """NPU optimized RMS Norm for Pangu model."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if is_npu_available:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
        else:
            # Fallback to CPU/GPU implementation
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states


class NPUPanguMLP(nn.Module):
    """NPU optimized MLP for Pangu model."""

    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x):
        if is_npu_available:
            gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
            intermediate_states = torch_npu.npu_swiglu(gate_up)
            return self.down_proj(intermediate_states)
        else:
            # Fallback to CPU/GPU implementation
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            intermediate_states = torch.nn.functional.silu(gate) * up
            return self.down_proj(intermediate_states)


def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """NPU optimized rotary position embedding."""
    if is_npu_available:
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
        return q_embed, k_embed
    else:
        # Fallback implementation
        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


def npu_flash_attention_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    is_causal,
    dropout=0.0,
    softmax_scale=None,
):
    """NPU optimized flash attention implementation."""
    if is_npu_available:
        # Use NPU flash attention
        return flash_attn_func(
            q=query_states,
            k=key_states,
            v=value_states,
            dropout_p=dropout,
            causal=is_causal,
            softmax_scale=softmax_scale,
        )
    else:
        # Fallback to regular attention
        import torch.nn.functional as F
        from einops import rearrange

        # Reshape for attention calculation
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        query_states = query_states.transpose(1, 2)  # [batch, seq_len, heads, dim]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim ** 0.5)

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_weights = attn_weights + attention_mask

        # Apply causal mask if needed
        if is_causal and seq_len > 1:
            causal_mask = torch.triu(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=attn_weights.device),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Softmax attention weights
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2)  # [batch, heads, seq_len, dim]
        return attn_output


def patch_pangu_for_npu(model):
    """Apply NPU optimizations to Pangu model."""
    if not is_npu_available:
        return model

    # Patch RMSNorm layers
    for name, module in model.named_modules():
        if hasattr(module, '__class__') and 'RMSNorm' in module.__class__.__name__:
            if not hasattr(module, 'weight'):  # Skip if already patched
                # Create NPU-optimized version
                npu_module = NPUPanguRMSNorm(
                    hidden_size=module.weight.shape[0],
                    eps=getattr(module, 'variance_epsilon', 1e-6)
                )
                npu_module.weight.data = module.weight.data.clone()

                # Replace in model
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, npu_module)
                else:
                    setattr(model, child_name, npu_module)

    print("Applied NPU optimizations to Pangu model")
    return model


def configure_npu_for_pangu():
    """Configure NPU settings for optimal Pangu performance."""
    if not is_npu_available:
        return

    # Enable NPU specific optimizations
    try:
        # Enable mixed precision for NPU
        torch.npu.set_compile_mode(jit_compile=True)

        # Configure NPU memory settings
        torch.npu.empty_cache()

        # Set NPU optimization level
        torch.npu.set_option("NPU_FUZZY_COMPILE_BLACKLIST", "100000")

        print("Configured NPU for Pangu model training")
    except Exception as e:
        print(f"Warning: Could not configure NPU optimizations: {e}")


def get_npu_memory_info():
    """Get NPU memory information."""
    if not is_npu_available:
        return {"available": False}

    try:
        total_memory = torch.npu.get_device_capability(0).total_memory
        allocated_memory = torch.npu.memory_allocated()
        free_memory = total_memory - allocated_memory

        return {
            "available": True,
            "total_memory": total_memory,
            "allocated_memory": allocated_memory,
            "free_memory": free_memory,
            "utilization": allocated_memory / total_memory
        }
    except Exception:
        return {"available": True, "error": "Could not retrieve memory info"}