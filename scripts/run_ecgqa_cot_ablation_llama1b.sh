#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src"

CURR_PY="${ROOT}/curriculum_learning.py"

# OpenTSLM wrapper to train (OpenTSLMSP or OpenTSLMFlamingo)
MODEL_TYPE="${MODEL_TYPE:-OpenTSLMSP}"

# ECG-only ablation settings
LLM_ID="meta-llama/Llama-3.2-1B"
FRACTIONS=(0.10 0.20 0.50)

mkdir -p "${ROOT}/logs"

for frac in "${FRACTIONS[@]}"; do
  echo ""
  echo "================================================================================"
  echo "  ECG-QA-CoT ablation | llm_id=${LLM_ID} | model=${MODEL_TYPE} | fraction=${frac}"
  echo "================================================================================"
  python "${CURR_PY}" \
    --model "${MODEL_TYPE}" \
    --llm_id "${LLM_ID}" \
    --ecg_only \
    --ecg_train_fraction "${frac}" \
    2>&1 | tee "${ROOT}/logs/ecgqa_cot_${MODEL_TYPE}_llama1b_fraction_${frac}.log"
done

echo ""
echo "Done: ECG-QA-CoT ablation for ${LLM_ID}."
