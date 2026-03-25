# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math
from typing import List
from opentslm.model_config import PATCH_SIZE

import torch.nn.functional as F

import torch

MAX_VALUE = 50_000
MIN_VALUE = -MAX_VALUE


def extend_time_series_to_match_patch_size_and_aggregate(
    batch, *, patch_size: int = PATCH_SIZE, normalize: bool = False
):
    """
    Pad variable-length series so each sample length is a multiple of *patch_size*.
    Optionally normalize each time series to have zero mean and unit variance.
    """

    for element in batch:
        # 1) pull out the list of (1D) time‑series
        ts_list = element["time_series"]

        # 2) convert each to a torch.Tensor (float)
        ts_tensors = [torch.as_tensor(ts, dtype=torch.float32) for ts in ts_list]

        # 3) normalize each time series if requested
        if normalize:
            normalized_tensors = []
            for ts in ts_tensors:
                mean = ts.mean()
                std = ts.std()
                if std > 1e-8:  # Avoid division by zero
                    ts_normalized = (ts - mean) / std
                else:
                    ts_normalized = ts - mean
                normalized_tensors.append(ts_normalized)
            ts_tensors = normalized_tensors

        # 4) find the longest series length
        max_len = max([ts.size(0) for ts in ts_tensors])

        # 5) round up to nearest multiple of patch_size
        padded_len = ((max_len + patch_size - 1) // patch_size) * patch_size

        # 6) pad (or trim) each series to padded_len
        padded = []
        for ts in ts_tensors:
            L = ts.size(0)
            if L < padded_len:
                pad_amt = padded_len - L
                ts = F.pad(ts, (0, pad_amt), mode="constant", value=0.0)
            else:
                ts = ts[:padded_len]
            padded.append(ts)

        # 7) stack into a single 2D tensor: (num_series, padded_len)
        element["time_series"] = torch.stack(padded, dim=0)

    return batch
