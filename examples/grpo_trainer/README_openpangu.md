# openPangu-Embedded-7B Integration with VERL

This document provides instructions for integrating the **openPangu-Embedded-7B** model from FreedomIntelligence into VERL for reinforcement learning tasks.

## 🚀 Backends Available

- **FSDP Backend**: Simple, works with any standard hardware (CPU/GPU)
- **Megatron Backend**: High-performance distributed training for large clusters
- **NPU Backend**: Optimized for Ascend NPU hardware with native acceleration

## Model Information

- **Model**: `FreedomIntelligence/openPangu-Embedded-7B`
- **Repository**: https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B
- **Architecture**: Pangu (Transformer-based)
- **Size**: 7 billion parameters

## Prerequisites

1. **Install VERL**: Follow the official VERL installation guide
2. **Install Dependencies**:
   ```bash
   pip install transformers torch vllm
   ```

3. **Prepare Dataset**: Have your training data in Parquet format
   - For example: GSM8K dataset for math reasoning
   - Expected columns: `prompt`, `response` (optional)

## Available Scripts

### 1. GRPO Training (Recommended)

**FSDP Backend (CPU/GPU):**
```bash
# Basic usage
./examples/grpo_trainer/run_openpangu-7b.sh

# With custom data
DATA_PATH=/path/to/your/data ./examples/grpo_trainer/run_openpangu-7b.sh

# Using YAML config
python -m verl.trainer.main_ppo --config-path examples/configs/openpangu_grpo_config.yaml
```

**Megatron Backend (High Performance):**
```bash
# Megatron GRPO with tensor parallelism
./examples/grpo_trainer/run_openpangu-7b_megatron.sh \
  megatron_config.tensor_model_parallel_size=4 \
  trainer.n_gpus_per_node=8
```

**NPU Backend (Ascend Hardware):**
```bash
# NPU-optimized GRPO training
./examples/grpo_trainer/run_openpangu-7b_npu.sh \
  trainer.n_gpus_per_node=16 \
  trainer.device=npu
```

### 2. PPO Training
```bash
# Basic usage
./examples/ppo_trainer/run_openpangu-7b.sh

# With custom parameters
./examples/ppo_trainer/run_openpangu-7b.sh trainer.n_gpus_per_node=8 trainer.total_epochs=30
```

## Configuration Options

### Key Parameters to Tune:

1. **Model Settings**:
   - `actor_rollout_ref.model.trust_remote_code=true`: Required for custom models
   - `actor_rollout_ref.model.enable_gradient_checkpointing=true`: Saves memory

2. **Training Parameters**:
   - `actor_rollout_ref.actor.optim.lr`: Learning rate (start with 1e-6)
   - `data.train_batch_size`: Based on your GPU memory
   - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: Adjust for OOM issues

3. **Memory Optimization**:
   - `actor_rollout_ref.rollout.gpu_memory_utilization`: VLLM memory usage (0.4-0.7)
   - `actor_rollout_ref.actor.fsdp_config.param_offload=true`: Offload parameters to CPU

## Example Use Cases

### Math Reasoning (GSM8K)
```bash
export DATA_PATH=$HOME/data/gsm8k
./examples/grpo_trainer/run_openpangu-7b.sh \
  data.train_batch_size=256 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  trainer.total_epochs=15
```

### Code Generation
```bash
export DATA_PATH=$HOME/data/codebench
./examples/grpo_trainer/run_openpangu-7b.sh \
  data.max_response_length=2048 \
  data.train_batch_size=128 \
  actor_rollout_ref.rollout.n=3 \
  trainer.total_epochs=10
```

### Custom Reward Model
```bash
./examples/grpo_trainer/run_openpangu-7b.sh \
  algorithm.use_kl_in_reward=false \
  trainer.project_name="custom_reward_openpangu"
```

## Multi-GPU Training

For multi-GPU training:

```bash
# 4 GPUs
./examples/grpo_trainer/run_openpangu-7b.sh trainer.n_gpus_per_node=4

# 8 GPUs with Megatron backend
./examples/grpo_trainer/run_openpangu-7b.sh \
  trainer.n_gpus_per_node=8 \
  actor_rollout_ref.actor.strategy=megatron \
  actor_rollout_ref.ref.strategy=megatron
```

## Troubleshooting

### Common Issues:

1. **Out of Memory (OOM)**:
   - Reduce `ppo_micro_batch_size_per_gpu`
   - Reduce `train_batch_size`
   - Enable `param_offload=true`
   - Reduce `gpu_memory_utilization`

2. **Model Loading Errors**:
   - Ensure `trust_remote_code=true` is set
   - Check if model is accessible from HuggingFace
   - Verify internet connection for initial model download

3. **Performance Issues**:
   - Use mixed precision training (default enabled)
   - Increase `tensor_model_parallel_size` for VLLM
   - Enable gradient checkpointing

## Advanced Integration

### Custom Model Adapter (Megatron Backend)

If you need Megatron backend integration for better performance:

1. Create model files in `verl/models/pangu/megatron/`
2. Implement `ParallelPanguForCausalLMRmPadPP` and related classes
3. Register in `verl/models/registry.py`:
   ```python
   "PanguForCausalLM": (
       "pangu",
       ("ParallelPanguForCausalLMRmPadPP", "ParallelPanguForValueRmPadPP", "ParallelPanguForCausalLMRmPad"),
   ),
   ```

### LoRA Fine-tuning

For memory-efficient training with LoRA:
```bash
./examples/grpo_trainer/run_openpangu-7b.sh \
  actor_rollout_ref.actor.lora.enabled=true \
  actor_rollout_ref.actor.lora.rank=64 \
  actor_rollout_ref.actor.lora.alpha=16 \
  actor_rollout_ref.actor.lora.target_modules=all-linear
```

## Monitoring

Track training progress:
```bash
# Enable wandb
./examples/grpo_trainer/run_openpangu-7b.sh trainer.logger='["console","wandb"]'

# Enable tensorboard
./examples/grpo_trainer/run_openpangu-7b.sh trainer.logger='["console","tensorboard"]'
```

## Citation

If you use openPangu-Embedded-7B with VERL in your research, please cite both the original model repository and VERL:

```bibtex
@model{openpangu_embedded_7b,
  title={openPangu-Embedded-7B},
  author={FreedomIntelligence},
  year={2024},
  url={https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B}
}

@software{verl,
  title={verl: Volcano Engine Reinforcement Learning for LLMs},
  author={Sheng et al.},
  year={2024},
  url={https://github.com/volcengine/verl}
}
```