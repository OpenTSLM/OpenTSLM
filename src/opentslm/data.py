# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import ast
from typing import Literal, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from opentslm.model_config import *

# ---------------------------
# Constants
# ---------------------------


# ---------------------------
# Core loader
# ---------------------------


def load_tsqa(
    split: Literal["train", "validation", "test"] = "train",
    *,
    max_samples: Optional[int] = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    EOS_TOKEN="",
):
    """Load the TSQA dataset with an explicit **train/validation/test** split.

    Args:
        split: which split to return.
        max_samples: optional cap on number of samples *after* splitting.
        val_frac: fraction (0–1) of the original data used for **validation**.
        test_frac: fraction (0–1) of the original data used for **test**.
        seed: RNG seed to make splits deterministic.
    Returns:
        ``datasets.Dataset`` with columns ["ts", "question", "answer"].
    """

    # 1) Load the single built‑in "train" split (≈ 7 k rows)
    ds_full = load_dataset("ChengsenWang/TSQA", split="train")

    # 2) First carve out the test split
    train_val, test = ds_full.train_test_split(test_size=test_frac, seed=seed).values()

    # 3) From the remaining data take validation
    train, val = train_val.train_test_split(
        test_size=val_frac / (1 - test_frac), seed=seed + 1
    ).values()

    # 4) Choose the requested split
    if split == "train":
        ds = train
    elif split in {"validation", "val"}:
        ds = val
    elif split == "test":
        ds = test
    else:
        raise ValueError("split must be 'train', 'validation', or 'test'")

    # 5) Optional size cap
    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    # 6) Pre-processing helper
    def _preprocess(ex):
        # --- normalise time‑series ---
        series = torch.tensor(ast.literal_eval(ex["Series"]), dtype=torch.float32)
        series = (series - series.mean()) / (series.std() + 1e-8)

        # --- clean Q/A and ensure EOS token ---
        question = ex["Question"].strip()
        answer = ex["Answer"].strip()
        if not answer.endswith(EOS_TOKEN):
            answer += EOS_TOKEN

        return {"ts": series, "question": question, "answer": answer}

    ds = ds.map(_preprocess)
    ds.set_format(type="torch", columns=["ts", "question", "answer"])
    return ds


# ---------------------------
# Collate + DataLoader helpers
# ---------------------------


def collate_fn(batch, *, patch_size: int = PATCH_SIZE):
    """Pad variable-length series so each sample length is a multiple of *patch_size*."""
    # pad length to the next multiple of patch_size among the batch
    max_len = max(ex["ts"].size(0) for ex in batch)
    max_len = ((max_len + patch_size - 1) // patch_size) * patch_size

    ts_list, qs, ans = [], [], []
    for ex in batch:
        ts = ex["ts"]
        if ts.size(0) < max_len:
            pad = max_len - ts.size(0)
            ts = torch.nn.functional.pad(ts, (0, pad), "constant", 0)
        else:
            ts = ts[:max_len]
        ts_list.append(ts)

        qs.append(ex["question"] + "\nAnswer:")
        ans.append(ex["answer"])

    return torch.stack(ts_list), qs, ans


def get_loader(
    split: Literal["train", "validation", "test"] = "train",
    *,
    batch_size: int = BATCH_SIZE,
    patch_size: int = PATCH_SIZE,
    max_samples: Optional[int] = None,
    shuffle: Optional[bool] = None,  # default True for train, False otherwise
    EOS_TOKEN="",
):
    """Convenience wrapper that returns a ``torch.utils.data.DataLoader`` for the requested split."""
    ds = load_tsqa(split=split, max_samples=max_samples, EOS_TOKEN=EOS_TOKEN)

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, patch_size=patch_size),
    )
