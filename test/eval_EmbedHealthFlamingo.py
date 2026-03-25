# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import json
from typing import List
from opentslm.time_series_datasets.TSQADataset import TSQADataset
from opentslm.time_series_datasets.monash.MonashSPO2QADataset import MonashSPO2QADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.model_config import (
    PATCH_SIZE,
    RESULTS_FILE,
)


# ---------------------------
# Device setup
# ---------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ---------------------------
# Model
# ---------------------------
model = OpenTSLMFlamingo(
    device=device,
    cross_attn_every_n_layers=1,
).to(device)


def merge_data_loaders(
    datasets: List[Dataset], shuffle: bool, batch_size: int, patch_size: int
) -> DataLoader:
    merged_ds = ConcatDataset(datasets)
    return DataLoader(
        merged_ds,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=patch_size
        ),
    )


QA_DATASET_CLASSES = [TSQADataset]

# ---------------------------
# Data loaders
# ---------------------------

test_loader = merge_data_loaders(
    [
        dataset_class(
            "test",
            EOS_TOKEN=model.get_eos_token(),
        )
        for dataset_class in QA_DATASET_CLASSES
    ],
    shuffle=False,
    batch_size=1,
    patch_size=PATCH_SIZE,
)


def _evaluate_test():
    """Run best model on test set and write prompt+generation+gold to JSONL."""
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test inference"):
            # batch is a List[Dict], same as in compute_loss/generate
            gens = model.generate(batch)  # returns List[str] of length len(batch)
            print(batch, gens, len(batch), len(gens))

            # collect each sample’s I/O
            for sample, gen in zip(batch, gens):
                results.append(
                    {
                        "pre_prompt": sample["pre_prompt"],
                        "time_series_text": sample["time_series_text"],
                        "post_prompt": sample["post_prompt"],
                        "generated": gen,
                        "gold": sample["answer"],
                    }
                )

    # write JSONL
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅  Test predictions saved to {RESULTS_FILE} (n={len(results)})")


if __name__ == "__main__":
    best_epoch = model.load_from_file("saved_models/4_bs_without_transformer_model.pt")
    if best_epoch is not None:
        print(f"Loaded best checkpoint from epoch {best_epoch} for test evaluation.")
    _evaluate_test()
