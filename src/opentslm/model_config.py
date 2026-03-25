# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

# ---------------------------
# Hyper‑parameters
# ---------------------------

BATCH_SIZE = 4
 
PATCH_SIZE = 4
NUM_EPOCHS = 20  # allow many but we will early‑stop
EARLY_STOP_PAT = 5  # stop if val loss hasn’t improved for this many epochs
LR_ENCODER = 2e-4
LR_PROJECTOR = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
WARMUP_FRAC = 0.03
MAX_SAMPLES = None  # set to an int for quick experiments
RESULTS_FILE = "test_predictions.jsonl"
EMBED_DIM = 128
ENCODER_OUTPUT_DIM = EMBED_DIM
TRANSFORMER_INPUT_DIM = EMBED_DIM
