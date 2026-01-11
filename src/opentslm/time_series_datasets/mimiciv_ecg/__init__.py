# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
MimicIV-ECG Dataset Module

This module provides tools for working with the MimicIV-ECG dataset for medical AI tasks.

Usage:
    from opentslm.time_series_datasets.mimiciv_ecg.MimicIVECGDataset import MimicIVECGDataset
    
    # Create dataset instance
    dataset = MimicIVECGDataset(split="test", EOS_TOKEN="", custom_prompts=["Does this ECG show atrial fibrillation?"])
    
    # Access samples
    sample = dataset[0]
"""

from .MimicIVECGDataset import MimicIVECGDataset
from .mimiciv_ecg_loader import (
    load_mimiciv_ecg_splits,
    get_mimiciv_ecg_path,
    load_mimiciv_ecg_answers,
)

__all__ = [
    "MimicIVECGDataset",
    "load_mimiciv_ecg_splits",
    "get_mimiciv_ecg_path",
    "load_mimiciv_ecg_answers",
]
