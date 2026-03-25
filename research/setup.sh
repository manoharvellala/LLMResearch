#!/bin/bash
# Run this after every RunPod resume to restore the Python environment.
set -e

# Set your HuggingFace token before running: export HF_TOKEN=your_token_here
# Or pass it inline: HF_TOKEN=your_token bash setup.sh
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set. Run: export HF_TOKEN=your_token" >&2
    exit 1
fi

pip install -q datasets matplotlib seaborn accelerate
pip install -q -e /workspace/transformers/

echo "Environment ready."
