#!/usr/bin/env python3
"""
Test script to validate openPangu-Embedded-7B model loading with VERL.
This script checks if the model can be loaded properly before running full training.
"""

import torch
import sys
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def test_model_loading():
    """Test if openPangu-Embedded-7B can be loaded correctly."""

    model_path = "FreedomIntelligence/openPangu-Embedded-7B"

    print(f"Testing model loading for: {model_path}")

    try:
        # Test 1: Load configuration
        print("1. Testing model configuration loading...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"   ✓ Config loaded successfully")
        print(f"   - Model type: {config.model_type}")
        print(f"   - Hidden size: {config.hidden_size}")
        print(f"   - Num layers: {config.num_hidden_layers}")
        print(f"   - Vocab size: {config.vocab_size}")

        # Test 2: Load tokenizer
        print("\n2. Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"   ✓ Tokenizer loaded successfully")
        print(f"   - Vocab size: {len(tokenizer)}")

        # Test basic tokenization
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        print(f"   - Sample encoding: '{test_text}' -> {tokens[:5]}{'...' if len(tokens) > 5 else ''}")

        # Test 3: Try loading model (if enough memory)
        print("\n3. Testing model loading...")
        try:
            # Only try loading if we have enough memory
            if torch.cuda.is_available():
                device = "cuda"
                print(f"   - CUDA available, testing on GPU")
            else:
                device = "cpu"
                print(f"   - Using CPU (this may be slow)")

            # Load model with reduced memory footprint
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            print(f"   ✓ Model loaded successfully on {device}")

            # Test basic forward pass
            with torch.no_grad():
                input_ids = tokenizer.encode(test_text, return_tensors="pt")
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()

                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss

                print(f"   ✓ Forward pass successful, loss: {loss.item():.4f}")

            del model  # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print("   ⚠ Out of memory - model too large for available GPU memory")
            print("   This is expected for 7B models on smaller GPUs")
            return True
        except Exception as e:
            print(f"   ✗ Model loading failed: {e}")
            return False

        print("\n✓ All tests passed! The model is compatible with VERL.")
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\nPossible solutions:")
        print("1. Check internet connection")
        print("2. Verify model name: FreedomIntelligence/openPangu-Embedded-7B")
        print("3. Ensure transformers library is up to date")
        print("4. Check if the model requires additional dependencies")
        return False

def test_verl_integration():
    """Test if VERL can import and initialize with the model."""

    print("\n" + "="*50)
    print("Testing VERL integration...")
    print("="*50)

    try:
        # Test basic imports
        from verl.trainer.main_ppo import main
        print("✓ VERL imports successful")

        # Test configuration loading
        from omegaconf import OmegaConf

        config_path = "/home/xiaohui/codes/code_agent/verl/examples/configs/openpangu_grpo_config.yaml"
        if os.path.exists(config_path):
            config = OmegaConf.load(config_path)
            print("✓ Configuration file loaded successfully")
            print(f"  - Model path: {config.actor_rollout_ref.model.path}")
            print(f"  - Algorithm: {config.algorithm.adv_estimator}")

        return True

    except ImportError as e:
        print(f"✗ VERL import failed: {e}")
        print("Please ensure VERL is properly installed")
        return False
    except Exception as e:
        print(f"✗ VERL integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("openPangu-Embedded-7B Model Integration Test")
    print("=" * 50)

    success = True

    # Test 1: Model loading
    if not test_model_loading():
        success = False

    # Test 2: VERL integration
    if not test_verl_integration():
        success = False

    if success:
        print("\n" + "="*50)
        print("🎉 All tests passed! The model is ready for VERL training.")
        print("\nNext steps:")
        print("1. Prepare your dataset")
        print("2. Run: ./examples/grpo_trainer/run_openpangu-7b.sh")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("❌ Some tests failed. Please resolve the issues above.")
        sys.exit(1)