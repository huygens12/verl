#!/bin/bash
set -x

# PPO training script for openPangu-Embedded-7B optimized for NPU (Ascend) hardware
# This script is specifically configured for NPU acceleration with Ascend chips

# Set up data paths - you can modify these based on your dataset
DATA_PATH=${DATA_PATH:-"$HOME/data/gsm8k"}
TRAIN_FILES="${DATA_PATH}/train.parquet"
VAL_FILES="${DATA_PATH}/test.parquet"}

# Model configuration
MODEL_PATH="FreedomIntelligence/openPangu-Embedded-7B"

python3 -m verl.trainer.main_ppo \
    algorithm=ppo \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=5e-8 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_coeff=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_ppo_openpangu_npu' \
    trainer.experiment_name='openpangu_7b_ppo_npu' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.device=npu $@

echo "PPO training with NPU backend completed for openPangu-Embedded-7B"