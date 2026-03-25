#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

python evaluate_tsqa.py "google/gemma-3-270m"
python evaluate_sleep_cot.py "google/gemma-3-270m"

python evaluate_tsqa.py "google/gemma-3-270m"
python evaluate_sleep_cot.py "google/gemma-3-270m"
python evaluate_har.py "google/gemma-3-1b-pt"

python evaluate_ecg_qa.py "google/gemma-3-270m"
python evaluate_ecg_qa.py "google/gemma-3-1b-pt"
python evaluate_ecg_qa.py "meta-llama/Llama-3.2-1B"
python evaluate_ecg_qa.py "meta-llama/Llama-3.2-3B"
