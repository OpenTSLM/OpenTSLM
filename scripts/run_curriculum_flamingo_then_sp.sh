#!/usr/bin/env bash
# Run full curriculum for OpenTSLMFlamingo, then OpenTSLMSP (soft prompt), with the same
# patch_size and shared extra arguments (e.g. --stages, --llm_id, --batch_size).
#
# Usage:
#   ./scripts/run_curriculum_flamingo_then_sp.sh PATCH_SIZE [extra curriculum_learning.py args...]
#
# Examples:
#   ./scripts/run_curriculum_flamingo_then_sp.sh 4
#   ./scripts/run_curriculum_flamingo_then_sp.sh 1 --llm_id meta-llama/Llama-3.2-1B
#   ./scripts/run_curriculum_flamingo_then_sp.sh 8 --stages stage3_cot stage4_sleep_cot stage5_ecg_cot
#
# To run Soft-Prompt first, swap the two "python" blocks at the bottom.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src"

PATCH_SIZE="${1:?Usage: $0 PATCH_SIZE [extra args passed to curriculum_learning.py ...]}"
shift

CURR_PY="${ROOT}/curriculum_learning.py"

run_model() {
  local model="$1"
  echo ""
  echo "================================================================================"
  echo "  ${model}  |  patch_size=${PATCH_SIZE}"
  echo "================================================================================"
  python "${CURR_PY}" \
    --model "${model}" \
    --patch_size "${PATCH_SIZE}" \
    "$@"
}

run_model OpenTSLMFlamingo "$@"
run_model OpenTSLMSP "$@"

echo ""
echo "Done: Flamingo and OpenTSLMSP (soft prompt) finished for patch_size=${PATCH_SIZE}."
