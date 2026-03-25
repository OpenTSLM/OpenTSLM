#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

# List of models to evaluate
models=(
    "google/gemma-3-270m"
    "google/gemma-3-1b-pt"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    
)

#echo "Starting evaluation of ${#models[@]} models on TSQA dataset..."
#for model in "${models[@]}"; do
#    echo "Evaluating $model on TSQA..."
#    python evaluate_tsqa.py "$model"
#done


echo "Starting evaluation of ${#models[@]} models on HAR dataset..."
for model in "${models[@]}"; do
    echo "Evaluating $model on PAMAP..."
    python evaluate_har.py "$model"
done

#echo "Starting evaluation of ${#models[@]} models on SleepEDF CoT dataset..."
#for model in "${models[@]}"; do
#    echo "Evaluating $model on SleepEDF CoT..."
#    python evaluate_sleep_cot.py "$model"
#done

echo "All evaluations completed!"