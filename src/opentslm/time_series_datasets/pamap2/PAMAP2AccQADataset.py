# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from datasets import Dataset
from typing import List, Tuple

import numpy as np
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.pamap2.PAMAP2Dataset import PAMAP2Dataset, ACTIVITIY_ID_DICT
from opentslm.time_series_datasets.pamap2.pamap2_loader import PAMAP2_DIR
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


TIME_SERIS_LABELS = [
    "The following is the accelerometer data on the x-axis",
    "The following is the accelerometer data on the y-axis",
    "The following is the accelerometer data on the z-axis",
]

MAIN_ACTITIVIES = [
    "lying",
    "sitting",
    "standing",
    "walking",
    "ascending stairs",
    "descending stairs",
    "running",
    "cycling",
    "nordic walking",
    "ironing",
    "vacuum cleaning",
    "rope jumping",
]


class PAMAP2AccQADataset(QADataset):
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        train = PAMAP2Dataset(
            [
                f"{PAMAP2_DIR}/Protocol/subject101.dat",
                f"{PAMAP2_DIR}/Protocol/subject102.dat",
                f"{PAMAP2_DIR}/Protocol/subject103.dat",
                f"{PAMAP2_DIR}/Protocol/subject104.dat",
                f"{PAMAP2_DIR}/Protocol/subject107.dat",
                f"{PAMAP2_DIR}/Protocol/subject108.dat",
                f"{PAMAP2_DIR}/Protocol/subject109.dat",
            ]
        )
        val = PAMAP2Dataset(
            [
                f"{PAMAP2_DIR}/Protocol/subject105.dat",
            ]
        )
        test = PAMAP2Dataset(
            [
                f"{PAMAP2_DIR}/Protocol/subject106.dat",
            ]
        )
        return train, val, test

    def _get_answer(self, row) -> str:
        return row["label"]

    def _get_pre_prompt(self, _row) -> str:
        return "You are given accelerometer data in all three dimensions, sampled at approximately 100Hz. Your task is to predict the person's activity."

    def _get_post_prompt(self, _row) -> str:
        activities = ", ".join(MAIN_ACTITIVIES)
        text = f"""

        Answer ONLY with the activity label.
        The following activities are possible: {activities}
        You MUST end your response with 'Answer: <class label>'
        """
        return text

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        series = torch.tensor(
            np.array(
                [
                    row["time_series"]["handAcc16_1"],
                    row["time_series"]["handAcc16_2"],
                    row["time_series"]["handAcc16_3"],
                ]
            ),
            dtype=torch.float32,
        )

        # Downsampling by 2x
        # Since the PAMAP dataset has 100Hz it results in around 50 Hz which should be fine for further processing
        series = series[:, ::2]
        # print(series.shape)

        # means = series.mean(dim=0, keepdim=True)  # shape: (n_series, 1)
        # stds = series.std(dim=0, keepdim=True)  # shape: (n_series, 1)
        # series = (series - means) / (stds + 1e-8)  # broadcasts to (n_series, length)

        return [
            TextTimeSeriesPrompt(time_series_label, time_series)
            for time_series_label, time_series in zip(
                TIME_SERIS_LABELS, series.tolist()
            )
        ]


if __name__ == "__main__":
    dataset = PAMAP2AccQADataset(split="train", EOS_TOKEN="")
    dataset_val = PAMAP2AccQADataset(split="validation", EOS_TOKEN="")
    dataset_test = PAMAP2AccQADataset(split="test", EOS_TOKEN="")

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

    print(len(dataset), len(dataset_val), len(dataset_test))
