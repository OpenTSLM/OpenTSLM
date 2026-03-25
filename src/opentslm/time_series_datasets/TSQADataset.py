# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import json
from typing import List, Tuple


from datasets import Dataset, load_dataset
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from torch.utils.data import DataLoader
import torch


TEST_FRAC = 0.1
VAL_FRAC = 0.1


def get_value_count(key: str, dataset: Dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=4
        ),
    )

    for value in dataloader:
        print(value)


class TSQADataset(QADataset):
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        # 1) Load the single built‑in "train" split (≈ 7 k rows)
        ds_full = load_dataset("ChengsenWang/TSQA", split="train")

        # 2) First carve out the test split
        train_val, test = ds_full.train_test_split(
            test_size=TEST_FRAC, seed=42
        ).values()

        # 3) From the remaining data take validation
        train, val = train_val.train_test_split(
            test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
        ).values()

        return train, val, test

    def _get_answer(self, row) -> str:
        return row["Answer"]

    def _get_pre_prompt(self, row) -> str:
        return row["Question"]

    def _get_post_prompt(self, row) -> str:
        # return "Answer:"
        return "Predict the " + row["Task"] + " Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        series = torch.tensor(json.loads(row["Series"]), dtype=torch.float32)

        means = series.mean(dim=0, keepdim=True)  # shape: (n_series, 1)
        stds = series.std(dim=0, keepdim=True)  # shape: (n_series, 1)
        series = (series - means) / (stds + 1e-8)  # broadcasts to (n_series, length)
        # TSQA has always only one time series
        # Make tensor indexing more robust
        mean_val = means.flatten()[0].item()
        std_val = stds.flatten()[0].item()
        return [TextTimeSeriesPrompt(f"This is the time series, it has mean {mean_val:.4f} and std {std_val:.4f}.", series.tolist())]


if __name__ == "__main__":
    train = TSQADataset("train", "")
    val = TSQADataset("validation", "")
    test = TSQADataset("test", "")

    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=4,
    #     shuffle=True,
    #     collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
    #         batch, patch_size=4
    #     ),
    # )

    from collections import Counter

    train_values = [
        (el[0], el[1] / len(train))
        for el in Counter(map(lambda x: x["post_prompt"], train)).items()
    ]
    print("train", train_values)
    val_values = [
        (el[0], el[1] / len(val))
        for el in Counter(map(lambda x: x["post_prompt"], val)).items()
    ]
    print("val", val_values)
    test_values = [
        (el[0], el[1] / len(test))
        for el in Counter(map(lambda x: x["post_prompt"], test)).items()
    ]
    print("test", test_values)
