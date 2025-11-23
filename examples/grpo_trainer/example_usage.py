#!/usr/bin/env python3
"""
Example usage of openPangu-Embedded-7B with VERL for RL tasks.
This script demonstrates how to set up and run training with sample data.
"""

import os
import sys
import subprocess
import json
import pandas as pd

def create_sample_data(output_dir: str = "sample_data"):
    """Create sample training data for demonstration."""

    os.makedirs(output_dir, exist_ok=True)

    # Sample math reasoning data (GSM8K style)
    train_data = [
        {
            "prompt": "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA:",
            "response": "Let's solve this step by step:\n\n1. Start with the initial number of trees: 15\n2. End with the final number of trees: 21\n3. The number of trees planted = final - initial = 21 - 15 = 6\n\nTherefore, the grove workers planted 6 trees today."
        },
        {
            "prompt": "Q: Leah had 32 apples. She gave away 14 apples to her friends. How many apples does Leah have left?\nA:",
            "response": "Let me solve this:\n\n1. Leah starts with 32 apples\n2. She gives away 14 apples\n3. Apples left = starting apples - apples given away = 32 - 14 = 18\n\nLeah has 18 apples left."
        },
        {
            "prompt": "Q: A bakery sells 120 cupcakes per day. If they are open 6 days a week, how many cupcakes do they sell in one week?\nA:",
            "response": "Here's how to solve this:\n\n1. Cupcakes sold per day: 120\n2. Days open per week: 6\n3. Total cupcakes per week = cupcakes per day × days per week = 120 × 6 = 720\n\nThe bakery sells 720 cupcakes in one week."
        }
    ]

    val_data = [
        {
            "prompt": "Q: If a book costs $15 and you buy 3 books, how much do you spend in total?\nA:",
            "response": "Calculation:\n1. Cost per book: $15\n2. Number of books: 3\n3. Total cost = $15 × 3 = $45\n\nYou spend $45 in total for 3 books."
        }
    ]

    # Save as Parquet files (required by VERL)
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_file = os.path.join(output_dir, "train.parquet")
    val_file = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)

    print(f"Created sample data:")
    print(f"  Train file: {train_file}")
    print(f"  Val file: {val_file}")

    return train_file, val_file

def run_training_example():
    """Run a complete training example."""

    print("Creating sample training data...")
    train_file, val_file = create_sample_data()

    print("\nSetting up training configuration...")

    # Basic training command
    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.train_batch_size=8",  # Small batch for demo
        "data.max_prompt_length=256",
        "data.max_response_length=512",
        "actor_rollout_ref.model.path=FreedomIntelligence/openPangu-Embedded-7B",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.n=2",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "trainer.logger='["console"]'",
        "trainer.project_name='verl_openpangu_demo'",
        "trainer.experiment_name='openpangu_7b_demo'",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=2",
        "trainer.test_freq=1",
        "trainer.total_epochs=3"
    ]

    print(f"\nTraining command:")
    print(" ".join(cmd))

    print("\n" + "="*60)
    print("READY TO START TRAINING")
    print("="*60)

    response = input("\nDo you want to run the training now? (y/n): ").lower().strip()

    if response == 'y' or response == 'yes':
        print("\nStarting training...")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Training completed successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Training failed with error: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
    else:
        print("\nTraining skipped.")
        print("You can run it manually with:")
        print(" ".join(cmd))

def show_tips():
    """Show useful tips for training."""

    print("\n" + "="*60)
    print("TRAINING TIPS")
    print("="*60)

    tips = [
        "1. MEMORY MANAGEMENT:",
        "   - Start with small batch sizes (2-8) if you get OOM errors",
        "   - Use gradient checkpointing to save memory",
        "   - Enable CPU offloading if needed: param_offload=true",
        "",
        "2. PERFORMANCE OPTIMIZATION:",
        "   - Use mixed precision (enabled by default)",
        "   - Increase tensor_model_parallel_size for VLLM speedup",
        "   - Set appropriate learning rate (1e-6 is good starting point)",
        "",
        "3. DATA PREPARATION:",
        "   - Ensure your data is in Parquet format",
        "   - Use appropriate prompt/response lengths",
        "   - Filter out overly long sequences",
        "",
        "4. MONITORING:",
        "   - Add wandb or tensorboard logging",
        "   - Save checkpoints regularly",
        "   - Monitor loss curves for convergence",
        "",
        "5. ALGORITHM SELECTION:",
        "   - GRPO: Good for general RL tasks (recommended)",
        "   - PPO: Classic RL algorithm, good stability",
        "   - Try different KL coefficients (0.001-0.1)"
    ]

    for tip in tips:
        print(tip)

if __name__ == "__main__":
    print("openPangu-Embedded-7B VERL Training Example")
    print("=" * 60)

    # Show prerequisites
    print("Prerequisites:")
    print("1. VERL installed: pip install verl")
    print("2. CUDA available (for GPU training)")
    print("3. Sufficient GPU memory (≥16GB recommended)")
    print("4. Internet connection for model download")
    print("5. Dependencies: transformers, vllm, pandas, pyarrow")

    # Check prerequisites
    try:
        import torch
        print(f"\n✓ PyTorch available: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPUs")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA not available - training will be on CPU (very slow)")
    except ImportError:
        print("✗ PyTorch not installed")
        sys.exit(1)

    try:
        import transformers
        print(f"✓ Transformers available: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
        sys.exit(1)

    # Show tips
    show_tips()

    # Run example
    run_training_example()

    print("\n" + "="*60)
    print("For more advanced configurations, see:")
    print("- README_openpangu.md")
    print("- run_openpangu-7b.sh")
    print("- openpangu_grpo_config.yaml")
    print("="*60)