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

"""Utilities for loading Pangu checkpoints into Megatron parallel models."""

import logging
import os
from typing import Dict, Optional, Union

import torch
from megatron.core import tensor_parallel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_huggingface_model(model_path: str, trust_remote_code: bool = True):
    """
    Load Pangu model from HuggingFace hub.

    Args:
        model_path: Path to model on HuggingFace hub or local path
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (model, tokenizer, config)
    """
    logger.info(f"Loading Pangu model from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load on CPU first
        low_cpu_mem_usage=True
    )

    # Get config
    config = model.config

    return model, tokenizer, config


def map_huggingface_to_megatron_keys(hf_key: str) -> Optional[str]:
    """
    Map HuggingFace parameter names to Megatron parameter names.

    Args:
        hf_key: HuggingFace parameter name

    Returns:
        Megatron parameter name or None if no mapping exists
    """
    # Embeddings
    if hf_key == "model.embed_tokens.weight":
        return "transformer.embed_tokens.weight"

    # Final layernorm
    if hf_key == "model.norm.weight":
        return "transformer.norm.weight"

    # Output head
    if hf_key == "lm_head.weight":
        return "lm_head.weight"

    # Layer mappings
    if "model.layers." in hf_key:
        layer_num = hf_key.split(".")[2]

        # Self attention
        if hf_key.endswith(".self_attn.qkv_proj.weight"):
            return f"transformer.layers.{layer_num}.self_attn.qkv_proj.weight"
        elif hf_key.endswith(".self_attn.o_proj.weight"):
            return f"transformer.layers.{layer_num}.self_attn.o_proj.weight"

        # MLP
        elif hf_key.endswith(".mlp.gate_up_proj.weight"):
            return f"transformer.layers.{layer_num}.mlp.gate_up_proj.weight"
        elif hf_key.endswith(".mlp.down_proj.weight"):
            return f"transformer.layers.{layer_num}.mlp.down_proj.weight"

        # Layer norms
        elif hf_key.endswith(".input_layernorm.weight"):
            return f"transformer.layers.{layer_num}.input_layernorm.weight"
        elif hf_key.endswith(".post_attention_layernorm.weight"):
            return f"transformer.layers.{layer_num}.post_attention_layernorm.weight"

    return None


def load_state_dict(
    hf_model_path: str,
    megatron_model,
    trust_remote_code: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Load and convert HuggingFace Pangu checkpoint to Megatron format.

    Args:
        hf_model_path: Path to HuggingFace model
        megatron_model: Target Megatron model
        trust_remote_code: Whether to trust remote code

    Returns:
        Dictionary of converted weights
    """
    logger.info("Loading HuggingFace checkpoint...")

    # Load HuggingFace model
    hf_model, _, _ = load_huggingface_model(hf_model_path, trust_remote_code)
    hf_state_dict = hf_model.state_dict()

    logger.info("Converting checkpoint to Megatron format...")

    converted_state_dict = {}

    for hf_key, hf_tensor in hf_state_dict.items():
        mg_key = map_huggingface_to_megatron_keys(hf_key)

        if mg_key is None:
            logger.warning(f"No mapping found for {hf_key}, skipping")
            continue

        if mg_key not in megatron_model.state_dict():
            logger.warning(f"Target key {mg_key} not found in model, skipping")
            continue

        converted_state_dict[mg_key] = hf_tensor
        logger.debug(f"Mapped {hf_key} -> {mg_key} (shape: {hf_tensor.shape})")

    logger.info(f"Converted {len(converted_state_dict)} parameters")

    return converted_state_dict


def load_checkpoint(
    model_path: str,
    megatron_model,
    trust_remote_code: bool = True,
    strict: bool = False,
):
    """
    Load checkpoint into Megatron model.

    Args:
        model_path: Path to HuggingFace model
        megatron_model: Target Megatron model
        trust_remote_code: Whether to trust remote code
        strict: Whether to enforce strict loading
    """
    # Convert and load weights
    converted_weights = load_state_dict(
        model_path,
        megatron_model,
        trust_remote_code=trust_remote_code
    )

    # Load into model
    missing_keys, unexpected_keys = megatron_model.load_state_dict(
        converted_weights,
        strict=strict
    )

    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")

    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")

    logger.info("Checkpoint loaded successfully!")

    return missing_keys, unexpected_keys


def save_checkpoint(
    megatron_model,
    save_path: str,
    epoch: int = 0,
    iteration: int = 0,
):
    """
    Save Megatron model checkpoint.

    Args:
        megatron_model: Megatron model to save
        save_path: Path to save checkpoint
        epoch: Current epoch
        iteration: Current iteration
    """
    os.makedirs(save_path, exist_ok=True)

    # Get state dict
    state_dict = megatron_model.state_dict()

    # Add metadata
    checkpoint = {
        "epoch": epoch,
        "iteration": iteration,
        "model_state_dict": state_dict,
    }

    # Save checkpoint
    checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch}_iter_{iteration}.pt")
    torch.save(checkpoint, checkpoint_path)

    logger.info(f"Checkpoint saved to {checkpoint_path}")


def initialize_megatron_model_from_hf(
    hf_model_path: str,
    megatron_model_class,
    megatron_config,
    trust_remote_code: bool = True,
):
    """
    Initialize Megatron model from HuggingFace checkpoint.

    Args:
        hf_model_path: Path to HuggingFace model
        megatron_model_class: Megatron model class
        megatron_config: Megatron configuration
        trust_remote_code: Whether to trust remote code

    Returns:
        Initialized Megatron model
    """
    # Load HuggingFace config
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=trust_remote_code)

    # Create Megatron model
    megatron_model = megatron_model_class(hf_config, megatron_config)

    # Load checkpoint
    load_checkpoint(hf_model_path, megatron_model, trust_remote_code=trust_remote_code)

    return megatron_model