#!/bin/bash
# Memory optimization script for CUDA training
# This sets PyTorch environment variables to reduce memory fragmentation

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Optional: Clear GPU memory before running
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Run your training script with all arguments passed through
python "$@"
