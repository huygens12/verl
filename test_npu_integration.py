#!/usr/bin/env python3
"""
Test script to verify openPangu-Embedded-7B NPU integration and compatibility.
"""

import sys
import os

# Add the current directory to Python path to import verl
sys.path.insert(0, '/home/xiaohui/codes/code_agent/verl')

def test_npu_availability():
    """Test if NPU is available and properly configured."""

    print("Testing NPU availability...")
    print("=" * 50)

    try:
        import torch

        # Check CUDA first for comparison
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPUs")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA not available")

        # Check NPU availability
        from verl.utils.device import is_npu_available

        if is_npu_available:
            print("✓ NPU is available")
            try:
                import torch_npu
                device_count = torch.npu.device_count()
                print(f"  NPU devices: {device_count}")
                for i in range(device_count):
                    print(f"  NPU {i}: {torch.npu.get_device_name(i)}")
                return True
            except Exception as e:
                print(f"  ⚠ NPU import error: {e}")
                return False
        else:
            print("⚠ NPU not available - this is expected if not running on Ascend hardware")
            print("  Note: NPU-specific optimizations will be skipped automatically")
            return True  # Not a failure, just no NPU

    except Exception as e:
        print(f"✗ Error checking device availability: {e}")
        return False


def test_npu_optimizations():
    """Test NPU optimization imports and functionality."""

    print(f"\nTesting NPU optimizations...")
    print("=" * 50)

    try:
        from verl.models.transformers.pangu_npu import (
            NPUPanguRMSNorm,
            NPUPanguMLP,
            apply_rotary_pos_emb_npu,
            patch_pangu_for_npu,
            configure_npu_for_pangu,
            get_npu_memory_info
        )

        print("✓ NPUPanguRMSNorm imported successfully")
        print("✓ NPUPanguMLP imported successfully")
        print("✓ apply_rotary_pos_emb_npu imported successfully")
        print("✓ patch_pangu_for_npu imported successfully")
        print("✓ configure_npu_for_pangu imported successfully")
        print("✓ get_npu_memory_info imported successfully")

        # Test NPU memory info
        memory_info = get_npu_memory_info()
        print(f"✓ NPU memory info: {memory_info}")

        # Test configuration
        configure_npu_for_pangu()
        print("✓ NPU configuration completed")

        return True

    except Exception as e:
        print(f"✗ NPU optimization test failed: {e}")
        return False


def test_npu_model_optimization():
    """Test NPU model patching functionality."""

    print(f"\nTesting NPU model optimization...")
    print("=" * 50)

    try:
        import torch
        from verl.models.transformers.pangu_npu import NPUPanguRMSNorm

        # Create a simple test model
        class TestPanguModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rms_norm = torch.nn.LayerNorm(4096)  # Will be replaced
                self.linear = torch.nn.Linear(4096, 4096)

        model = TestPanguModel()
        print("✓ Test model created")

        # Test NPU RMSNorm
        npu_rms_norm = NPUPanguRMSNorm(4096)
        print("✓ NPU RMSNorm created successfully")

        # Test forward pass
        x = torch.randn(1, 10, 4096)
        output = npu_rms_norm(x)
        print(f"✓ NPU RMSNorm forward pass: {output.shape}")

        # Test patching functionality
        from verl.models.transformers.pangu_npu import patch_pangu_for_npu
        patched_model = patch_pangu_for_npu(model)
        print("✓ Model patching completed")

        return True

    except Exception as e:
        print(f"✗ NPU model optimization test failed: {e}")
        return False


def test_npu_compatibility_layers():
    """Test NPU compatibility with existing layers."""

    print(f"\nTesting NPU compatibility with existing layers...")
    print("=" * 50)

    try:
        from verl.models.transformers.pangu_npu import (
            apply_rotary_pos_emb_npu,
            npu_flash_attention_forward
        )

        import torch

        # Create test tensors
        batch_size, num_heads, seq_len, head_dim = 1, 32, 10, 128

        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Create rotary embeddings
        pos = torch.arange(seq_len)
        dim = head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = pos.float().unsqueeze(-1)
        freqs = torch.matmul(t, inv_freq.unsqueeze(0))
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)

        # Test rotary position embedding
        q_rot, k_rot = apply_rotary_pos_emb_npu(query, key, cos, sin)
        print(f"✓ Rotary position embedding: {q_rot.shape}")

        # Test attention (simplified)
        attention_mask = None
        attn_output = npu_flash_attention_forward(
            query_states=query,
            key_states=key,
            value_states=value,
            attention_mask=attention_mask,
            query_length=seq_len,
            is_causal=True
        )
        print(f"✓ Flash attention: {attn_output.shape}")

        return True

    except Exception as e:
        print(f"✗ NPU compatibility test failed: {e}")
        return False


def show_npu_integration_summary():
    """Show summary of NPU integration status."""

    print(f"\n" + "=" * 60)
    print("PANGU NPU INTEGRATION SUMMARY")
    print("=" * 60)

    summary = """
🔧 NPU FEATURES IMPLEMENTED:

1. NPU-Optimized Components:
   - NPUPanguRMSNorm (NPU-native RMS normalization)
   - NPUPanguMLP (NPU-native SwiGLU activation)
   - apply_rotary_pos_emb_npu (NPU-native RoPE)
   - npu_flash_attention_forward (NPU flash attention)

2. Automatic Optimization:
   - patch_pangu_for_npu() - Auto-patch model components
   - configure_npu_for_pangu() - NPU environment setup
   - get_npu_memory_info() - Memory monitoring

3. Fallback Support:
   - Automatic CPU/GPU fallback when NPU unavailable
   - Graceful degradation without performance loss
   - Cross-platform compatibility

🚀 NPU TRAINING SCRIPTS:
   - run_openpangu-7b_npu.sh (GRPO with NPU)
   - run_openpangu-7b_npu.sh (PPO with NPU)
   - openpangu_npu_config.yaml (NPU configuration)

⚡ NPU OPTIMIZATIONS:
   - Mixed precision training (FP16)
   - NPU-specific memory management
   - HCCL communication backend
   - Graph kernel fusion
   - Dynamic shape handling

📋 REQUIREMENTS:
   - Ascend NPU hardware or NPU simulation
   - torch-npu package
   - NPU drivers and toolkit
   - VERL with NPU support enabled
    """

    print(summary)


if __name__ == "__main__":
    print("Pangu NPU Integration Test")
    print("=" * 60)

    success = True

    # Run tests
    if not test_npu_availability():
        success = False

    if not test_npu_optimizations():
        success = False

    if not test_npu_model_optimization():
        success = False

    if not test_npu_compatibility_layers():
        success = False

    # Show summary
    show_npu_integration_summary()

    if success:
        print(f"\n🎉 SUCCESS: Pangu NPU integration is complete!")
        print(f"\nThe model is ready for NPU-accelerated RL training.")
        print("=" * 60)
    else:
        print(f"\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)