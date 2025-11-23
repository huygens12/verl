#!/usr/bin/env python3
"""
Download openPangu-Embedded-7B model for offline usage.
Run this on a machine with internet access, then transfer to offline environment.
"""

import os
import sys
import argparse
from pathlib import Path

def download_model(model_name: str, output_dir: str):
    """Download model from HuggingFace for offline use."""

    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, AutoConfig
        print("✅ huggingface_hub and transformers imported successfully")
    except ImportError:
        print("❌ Please install: pip install huggingface_hub transformers")
        return False

    print(f"📦 Downloading {model_name}...")
    print(f"📁 Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Download entire model repository
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False,  # Important for offline transfer
            resume_download=True
        )
        print(f"✅ Model downloaded to {output_dir}")

        # Verify the download
        print(f"🔍 Verifying downloaded model...")

        # Test config loading
        config_path = os.path.join(output_dir, "config.json")
        if os.path.exists(config_path):
            config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
            print(f"   ✅ Config loaded: {config.model_type}")
            print(f"   ✅ Hidden size: {config.hidden_size}")
            print(f"   ✅ Layers: {config.num_hidden_layers}")
        else:
            print(f"   ⚠ config.json not found")

        # Test tokenizer loading
        try:
            tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
            print(f"   ✅ Tokenizer loaded: {len(tokenizer)} tokens")
        except Exception as e:
            print(f"   ⚠ Tokenizer test failed: {e}")

        # Check for model files
        model_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json"
        ]

        missing_files = []
        for file in model_files:
            if not os.path.exists(os.path.join(output_dir, file)):
                missing_files.append(file)

        if missing_files:
            print(f"   ⚠ Missing files: {missing_files}")
            print("   ⚠ Model may not work completely offline")
        else:
            print(f"   ✅ All required files present")

        print(f"🎉 Model download completed successfully!")
        print(f"📊 Size: {sum(f.stat().st_size for f in Path(output_dir).rglob('*') if f.is_file()) / 1024**3:.1f} GB")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def create_offline_readme(output_dir: str):
    """Create README with offline setup instructions."""

    readme_content = f"""# openPangu-Embedded-7B Offline Model

This model was downloaded for offline usage from FreedomIntelligence/openPangu-Embedded-7B.

## Usage with VERL

Use this local path in your VERL training scripts:

```bash
./examples/grpo_trainer/run_openpangu-7b.sh \\
  actor_rollout_ref.model.path={output_dir}
```

## Files Included

"""

    # List all files
    for file in sorted(Path(output_dir).rglob('*')):
        if file.is_file():
            size_mb = file.stat().st_size / 1024**2
            readme_content += f"- `{file.relative_to(output_dir)}` ({size_mb:.1f} MB)\n"

    readme_content += """
## Environment Setup

```bash
# Set HuggingFace cache to avoid network calls
export TRANSFORMERS_CACHE={output_dir}
export HF_HOME={output_dir}
export OFFLINE_MODE=1
```

## Verification

Test the offline setup:
```bash
python3 /path/to/verl/scripts/test_offline_model.py {output_dir}
```
"""

    readme_path = os.path.join(output_dir, "OFFLINE_README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"📄 Created {readme_path}")

def main():
    parser = argparse.ArgumentParser(description="Download openPangu model for offline use")
    parser.add_argument(
        "--model-name",
        default="FreedomIntelligence/openPangu-Embedded-7B",
        help="Model name to download"
    )
    parser.add_argument(
        "--output-dir",
        default="./openPangu-Embedded-7B-offline",
        help="Output directory for downloaded model"
    )

    args = parser.parse_args()

    print("🚀 openPangu Model Downloader (Offline Mode)")
    print("=" * 60)

    success = download_model(args.model_name, args.output_dir)

    if success:
        create_offline_readme(args.output_dir)

        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Model ready for offline use")
        print("=" * 60)
        print(f"📁 Model location: {args.output_dir}")
        print(f"📋 Next steps:")
        print(f"   1. Transfer this directory to your offline machine")
        print(f"   2. Update VERL scripts to use: actor_rollout_ref.model.path={args.output_dir}")
        print(f"   3. Set environment: export TRANSFORMERS_CACHE={args.output_dir}")
        print("=" * 60)

    else:
        print("\n❌ FAILED! Please check the error above and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()