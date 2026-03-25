#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

# Usage:
#   bash run_all_memory.sh [--device cuda:0] [--results_csv memory_use.csv]

DEVICE_ARG=${1:-}
RESULTS_CSV=${2:-}

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python

# Models and datasets to test
MODELS=(
    "OpenTSLMSP"
    "OpenTSLMFlamingo"
)

DATASETS=(
  "TSQADataset"
  "HARCoTQADataset"
  "SleepEDFCoTQADataset"
  "ECGQACoTQADataset"
)

# LLM IDs
LLM_IDS=(
  "meta-llama/Llama-3.2-1B"
  "meta-llama/Llama-3.2-3B"
  "google/gemma-3-270m"
  "google/gemma-3-1b"
)

# SimulationQADataset parameters
SIMULATION_LENGTHS=(10 100 1000 10000)
SIMULATION_NUM_SERIES=(1 2 3 4 5)

RESULTS_FLAG="--results_csv ${RESULTS_CSV:-$REPO_DIR/memory_use.csv}"

if [[ -n "$DEVICE_ARG" ]]; then
  DEVICE_FLAG="--device ${DEVICE_ARG#--device }"
else
  DEVICE_FLAG=""
fi

echo "Writing results to: ${RESULTS_FLAG#--results_csv }"

for llm in "${LLM_IDS[@]}"; do
  for model in "${MODELS[@]}"; do
    # Run standard datasets
    for dataset in "${DATASETS[@]}"; do
      echo "[RUN] llm_id=$llm model=$model dataset=$dataset"
      set +e
      $PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "$llm" --model "$model" --dataset "$dataset" $DEVICE_FLAG $RESULTS_FLAG
      status=$?
      set -e
      if [[ $status -ne 0 ]]; then
        echo "[ERROR] Failed for llm_id=$llm model=$model dataset=$dataset (exit $status)"
      fi
    done
    
    # Run SimulationQADataset with all combinations of length and num_series
    for length in "${SIMULATION_LENGTHS[@]}"; do
      for num_series in "${SIMULATION_NUM_SERIES[@]}"; do
        echo "[RUN] llm_id=$llm model=$model dataset=SimulationQADataset length=$length num_series=$num_series"
        set +e
        $PYTHON "$REPO_DIR/get_memory_use.py" -llm_id "$llm" --model "$model" --dataset "SimulationQADataset" --length "$length" --num_series "$num_series" $DEVICE_FLAG $RESULTS_FLAG
        status=$?
        set -e
        if [[ $status -ne 0 ]]; then
          echo "[ERROR] Failed for llm_id=$llm model=$model dataset=SimulationQADataset length=$length num_series=$num_series (exit $status)"
        fi
      done
    done
  done
done

echo "All runs completed."

