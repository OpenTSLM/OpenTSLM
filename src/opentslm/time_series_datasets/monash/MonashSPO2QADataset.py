# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from typing import List, Literal, Optional, Tuple
from datasets import Dataset
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.monash.MonashDataset import MonashDataset
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
import torch
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm


TIME_SERIS_LABELS = ["The following is PPG data", "The following is ECG data"]


class MonashSPO2QADataset(QADataset):
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        train = MonashDataset(
            _data_dir="monash_datasets",
            data_name="BIDMC32SpO2/BIDMC32SpO2_TRAIN",
        )
        test = MonashDataset(
            _data_dir="monash_datasets",
            data_name="BIDMC32SpO2/BIDMC32SpO2_TEST",
        )

        train_size = int(len(train) * 0.9)
        val_size = len(train) - train_size

        train, val = random_split(
            train, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        return train, val, test

    def _get_answer(self, row) -> str:
        return str(round(row["answer"], 2))

    def _get_pre_prompt(self, _row) -> str:
        return "You are given PPG and ECG data. Your task is to predict the average blood oxygen saturation over the given the 32 second period."

    def _get_post_prompt(self, _row) -> str:
        return "Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        if len(row["time_series"][0]) != len(TIME_SERIS_LABELS):
            raise RuntimeError(
                "question labels and time series from the data must be of the same length"
            )

        # TODO normalize

        return [
            TextTimeSeriesPrompt(time_series_label, time_series)
            for time_series_label, time_series in zip(
                TIME_SERIS_LABELS, row["time_series"][0]
            )
        ]


if __name__ == "__main__":
    dataset = MonashSPO2QADataset(split="train", EOS_TOKEN="")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=4
        ),
    )

    for batch in tqdm(dataloader):
        print(batch)
