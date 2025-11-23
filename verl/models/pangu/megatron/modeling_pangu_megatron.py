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

"""PyTorch Pangu model with Megatron tensor parallelism."""

from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core import ModelParallelConfig, mpu, parallel_state, tensor_parallel
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from verl.utils.device import get_device_name
from verl.utils.megaton import sequence_parallel as sp_utils
from verl.utils.megaton import tensor_parallel as tp_utils
from verl.utils.megatron_utils import TransformerConfig, convert_config

from .layers import ParallelPanguDecoderLayer, ParallelPanguDecoderLayerRmPad, ParallelPanguRMSNorm


class PanguConfig(PretrainedConfig):
    """
    Configuration class for Pangu model.
    """

    model_type = "pangu"

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class ParallelPanguModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PanguDecoderLayer`]
    """

    def __init__(self, config: PanguConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            assert embedding_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(embedding_kwargs, megatron_config)

        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            **embedding_kwargs
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [ParallelPanguDecoderLayer(config, megatron_config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer norm
        self.norm = ParallelPanguRMSNorm(config, megatron_config)

        # Gradient checkpointing
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple:

        # Handle input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Dropout
        hidden_states = inputs_embeds

        # Get sequence length
        seq_length = hidden_states.size()[1]

        # Initialize attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Prepare past key values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Forward through layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)

        # Final layernorm
        hidden_states = self.norm(hidden_states)

        # Add final hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        return (hidden_states, next_cache, all_hidden_states, all_self_attns)


class ParallelPanguForCausalLMRmPad(nn.Module):
    """
    Pangu model for causal language modeling with remove padding support.
    """

    def __init__(self, config: PanguConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.transformer = ParallelPanguModel(config, megatron_config)

        # Language modeling head
        lm_head_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(lm_head_kwargs, megatron_config)

        self.lm_head = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=True,  # For language modeling, we need full vocab
            **lm_head_kwargs
        )

        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:

        # Forward through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Language modeling head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs[1] if len(transformer_outputs) > 1 else None,
            hidden_states=transformer_outputs[2] if len(transformer_outputs) > 2 else None,
            attentions=transformer_outputs[3] if len(transformer_outputs) > 3 else None,
        )


class ParallelPanguForCausalLMRmPadPP(ParallelPanguForCausalLMRmPad):
    """Pangu model for causal language modeling with PP (Path Parallel) support."""
    pass


class ParallelPanguForValueRmPadPP(nn.Module):
    """
    Pangu model for value head with remove padding and PP support.
    """

    def __init__(self, config: PanguConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.transformer = ParallelPanguModel(config, megatron_config)

        # Value head
        value_head_kwargs = tp_utils.get_default_kwargs_for_parallel_linear()
        if megatron_config is not None:
            tp_utils.update_kwargs_with_config(value_head_kwargs, megatron_config)

        self.value_head = tensor_parallel.RowParallelLinear(
            config.hidden_size,
            1,  # Single value output
            bias=False,
            input_is_parallel=True,
            **value_head_kwargs
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Forward through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Value head
        values = self.value_head(hidden_states).squeeze(-1)

        return values