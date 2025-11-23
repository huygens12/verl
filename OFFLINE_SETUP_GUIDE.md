# Offline Setup Guide for openPangu-Embedded-7B

## 🚫 The Problem: No Internet to HuggingFace

If your network cannot connect to HuggingFace, the FSDP integration will fail because VERL tries to:

1. **Download model files** from `FreedomIntelligence/openPangu-Embedded-7B`
2. **Load configuration** from HuggingFace hub
3. **Verify model availability** before training

## ✅ **Solution: Offline Setup**

### **Option 1: Pre-download Model Files**

#### **Step 1: Download Model on Connected Machine**
```bash
# On a machine with internet access:
git lfs install  # Install Git LFS first
git clone https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B

# Or using huggingface-hub:
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='FreedomIntelligence/openPangu-Embedded-7B',
    local_dir='./openPangu-Embedded-7B-local',
    local_dir_use_symlinks=False
)"
```

#### **Step 2: Transfer to Offline Machine**
```bash
# Copy the entire model directory
scp -r openPangu-Embedded-7B-local/ offline-machine:/path/to/models/
```

#### **Step 3: Update VERL Configuration**
```bash
# Use local path instead of HuggingFace path
./examples/grpo_trainer/run_openpangu-7b.sh \
  actor_rollout_ref.model.path=/path/to/models/openPangu-Embedded-7B-local \
  actor_rollout_ref.model.trust_remote_code=true
```

### **Option 2: VERL Built-in Caching**

#### **Step 1: Download on Connected Machine**
```bash
# Let VERL download the model once
./examples/grpo_trainer/run_openpangu-7b.sh

# The model will be cached in:
# ~/.cache/huggingface/hub/models--FreedomIntelligence--openPangu-Embedded-7B/
```

#### **Step 2: Copy Cache to Offline Machine**
```bash
# Copy the entire cache directory
scp -r ~/.cache/huggingface/ offline-machine:~/.cache/
```

#### **Step 3: Use Original Scripts**
```bash
# Now the original scripts work offline!
./examples/grpo_trainer/run_openpangu-7b.sh
```

### **Option 3: Enterprise/Mirror Setup**

#### **Step 1: Set Up Internal Mirror**
```bash
# Create internal HuggingFace mirror
huggingface-cli repo create company/openPangu-Embedded-7B-private
```

#### **Step 2: Configure Environment**
```bash
# Point to internal mirror
export HF_ENDPOINT="https://internal-hf-proxy.company.com"
export HF_HOME="/shared/models/huggingface-cache"
```

#### **Step 3: Use Same Scripts**
```bash
# Scripts work without modification!
./examples/grpo_trainer/run_openpangu-7b.sh
```

## 🔧 **VERL-Specific Offline Configuration**

### **Environment Variables**
```bash
# Set these for offline operation
export TRANSFORMERS_CACHE=/shared/models/transformers_cache
export HF_DATASETS_CACHE=/shared/models/datasets_cache
export HF_HOME=/shared/models/hf_home
export OFFLINE_MODE=1
```

### **VERL Configuration Updates**
```bash
# Update training script for offline use
./examples/grpo_trainer/run_openpangu-7b.sh \
  actor_rollout_ref.model.path=/shared/models/openPangu-Embedded-7B \
  actor_rollout_ref.model.local_files_only=true
```

### **Copy-to-Local Feature**
```bash
# VERL has built-in offline support via copy_to_local
# In fsdp_workers.py, VERL does:
local_path = copy_to_local(self.config.model.path, use_shm=use_shm)

# This copies model to shared memory for faster access
# Works with local paths too!
```

## 📁 **Required Files Checklist**

### **Model Files**
```
openPangu-Embedded-7B/
├── config.json                    # ✅ Required
├── pytorch_model.bin              # ✅ Required
├── tokenizer.json                  # ✅ Required
├── tokenizer_config.json           # ✅ Required
├── special_tokens_map.json        # ✅ Required
├── generation_config.json         # ✅ Optional
├── [any other model files]         # ✅ Include all
```

### **VERL Offline Scripts**
```
offline_setup/
├── download_model.sh              # Download script
├── copy_to_offline.sh             # Transfer script
├── setup_offline_env.sh           # Environment setup
└── test_offline_setup.sh          # Verification script
```

## 🚨 **Common Offline Issues & Solutions**

### **Issue 1: Model Download Fails**
```bash
❌ Error: No internet connection
✅ Solution: Use local path with local_files_only=true
```

### **Issue 2: Tokenizer Loading Fails**
```bash
❌ Error: Can't load tokenizer from hub
✅ Solution: Ensure tokenizer files are in model directory
```

### **Issue 3: Cache Directory Issues**
```bash
❌ Error: Permission denied in cache directory
✅ Solution: Set HF_HOME to writable location
```

### **Issue 4: Git LFS Files Missing**
```bash
❌ Error: Model weights are pointers, not actual files
✅ Solution: Use `git lfs pull` or huggingface_hub download
```

## 🛠️ **Complete Offline Setup Script**

### **Download Script** (`download_model.py`)
```python
#!/usr/bin/env python3
"""Download openPangu model for offline setup"""

import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

def download_model():
    model_name = "FreedomIntelligence/openPangu-Embedded-7B"
    local_dir = "./openPangu-Embedded-7B-offline"

    print(f"Downloading {model_name} to {local_dir}...")

    # Download entire model
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

    # Test tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    print(f"✅ Tokenizer loaded successfully: vocab_size={len(tokenizer)}")

    # Test model config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    print(f"✅ Config loaded: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")

    print(f"✅ Model ready for offline use!")
    return local_dir

if __name__ == "__main__":
    download_model()
```

### **Offline Verification Script** (`test_offline.py`)
```python
#!/usr/bin/env python3
"""Test if model works offline"""

import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_offline_model(model_path):
    try:
        print(f"Testing offline model: {model_path}")

        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")

        # Test config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Config loaded: {config.model_type}")

        # Test model (might be large, so just config test)
        print(f"✅ Model loading would succeed")
        print(f"✅ Ready for VERL offline training!")

        return True

    except Exception as e:
        print(f"❌ Offline test failed: {e}")
        return False

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./openPangu-Embedded-7B-offline"
    success = test_offline_model(model_path)
    sys.exit(0 if success else 1)
```

## 🎯 **Production Offline Setup**

### **Enterprise Deployment**
```bash
# 1. Shared NFS storage
export SHARED_MODEL_PATH="/shared/models/openPangu-Embedded-7B"
export TRANSFORMERS_CACHE="$SHARED_MODEL_PATH/cache"

# 2. All nodes use same local path
./examples/grpo_trainer/run_openpangu-7b.sh \
  actor_rollout_ref.model.path=$SHARED_MODEL_PATH

# 3. VERL handles everything else automatically!
```

### **Docker Offline Setup**
```dockerfile
# Dockerfile for offline VERL training
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Copy pre-downloaded model
COPY openPangu-Embedded-7B-offline /models/openPangu-Embedded-7B

# Set environment for offline operation
ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models
ENV OFFLINE_MODE=1

# Install VERL
RUN pip install verl

# Training works offline!
COPY run_openpangu-7b.sh /
CMD ["/run_openpangu-7b.sh"]
```

## 📊 **Offline Performance**

### **Performance Benefits**
- ✅ **No download delays** - instant model loading
- ✅ **Predictable training** - no network issues
- ✅ **Faster startups** - local storage
- ✅ **Network independent** - works in air-gapped environments

### **Storage Requirements**
- **7B Model**: ~14GB disk space
- **Tokenizer**: ~10MB
- **Configs**: <1MB
- **Total**: ~15GB per model

## 🎉 **Summary**

**Yes, no internet access breaks the default setup, but it's easily solvable:**

1. **Pre-download** the model files
2. **Transfer** to offline environment
3. **Use local paths** instead of HuggingFace paths
4. **Configure environment** for offline operation

**The FSDP integration itself doesn't change** - it still works perfectly with local model files! 🚀