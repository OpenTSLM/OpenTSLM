#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src"

CURR_PY="${ROOT}/curriculum_learning.py"

# OpenTSLM wrapper to train (OpenTSLMSP or OpenTSLMFlamingo)
# Default to Flamingo for this ablation.
MODEL_TYPE="${MODEL_TYPE:-OpenTSLMFlamingo}"

# Distributed (multi-GPU) settings
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TORCHRUN="${TORCHRUN:-torchrun}"

# Validation frequency (see curriculum_learning.py --val_every_n_epochs)
VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-2}"

# ECG-only ablation settings
LLM_ID="meta-llama/Llama-3.2-3B"
FRACTIONS=(0.10 0.20 0.50)

mkdir -p "${ROOT}/logs"

for frac in "${FRACTIONS[@]}"; do
  echo ""
  echo "================================================================================"
  echo "  ECG-QA-CoT ablation | llm_id=${LLM_ID} | model=${MODEL_TYPE} | fraction=${frac} | gpus=${NPROC_PER_NODE} | val_every=${VAL_EVERY_N_EPOCHS}"
  echo "================================================================================"
  "${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" "${CURR_PY}" \
    --model "${MODEL_TYPE}" \
    --llm_id "${LLM_ID}" \
    --batch_size 1 \
    --val_every_n_epochs "${VAL_EVERY_N_EPOCHS}" \
    --ecg_only \
    --ecg_train_fraction "${frac}" \
    2>&1 | tee "${ROOT}/logs/ecgqa_cot_${MODEL_TYPE}_llama3b_fraction_${frac}.log"
done

echo ""
echo "Done: ECG-QA-CoT ablation for ${LLM_ID}."
