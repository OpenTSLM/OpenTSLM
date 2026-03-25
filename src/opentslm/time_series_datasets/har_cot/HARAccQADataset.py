# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from datasets import Dataset
from typing import List, Tuple, Literal

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.har_cot.har_cot_loader import load_har_cot_splits
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

# Simple axis labels (no stats in-text for Acc variant)
TIME_SERIES_LABELS = [
    "The following is the accelerometer data on the x-axis",
    "The following is the accelerometer data on the y-axis",
    "The following is the accelerometer data on the z-axis",
]


class HARAccQADataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(
            split, EOS_TOKEN, format_sample_str, time_series_format_function
        )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load the HAR dataset splits using the har_cot_loader, but ignore rationales.

        Returns:
            Tuple of (train, validation, test) datasets
        """
        return load_har_cot_splits()

    def _get_answer(self, row) -> str:
        return row["label"]

    def _get_pre_prompt(self, _row) -> str:
        return "You are given accelerometer data in all three dimensions, sampled over time. Your task is to predict the person's activity."

    def _get_post_prompt(self, _row) -> str:
        activities = ", ".join(self.get_labels())
        text = f"""
Instructions:
- Begin by analyzing the time series without assuming a specific label.
- Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any class label until the final sentence.
The following activities (class labels) are possible: {activities}
- You MUST end your response with "Answer: <class label>"
"""
        return text

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the time series data into a list of TextTimeSeriesPrompt objects.
        Does not normalize the data.
        """
        series = torch.tensor(
            [
                row["x_axis"],
                row["y_axis"],
                row["z_axis"],
            ],
            dtype=torch.float32,
        )

        return [
            TextTimeSeriesPrompt(time_series_label, time_series)
            for time_series_label, time_series in zip(
                TIME_SERIES_LABELS, series.tolist()
            )
        ]

    @staticmethod
    def get_labels() -> List[str]:
        return [
            "biking",
            "lying",
            "running",
            "sitting",
            "standing",
            "walking",
            "walking_down",
            "walking_up",
        ]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["label"] = row["label"]
        sample["x_axis"] = row["x_axis"]
        sample["y_axis"] = row["y_axis"]
        sample["z_axis"] = row["z_axis"]
        return sample


if __name__ == "__main__":
    dataset = HARAccQADataset(split="train", EOS_TOKEN="")
    dataset_val = HARAccQADataset(split="validation", EOS_TOKEN="")
    dataset_test = HARAccQADataset(split="test", EOS_TOKEN="")

    print(
        f"Dataset sizes: Train: {len(dataset)}, Validation: {len(dataset_val)}, Test: {len(dataset_test)}"
    )

    dataloader = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=4
        ),
    )

    for batch in tqdm(dataloader, total=1):
        print("Batch keys:", batch[0].keys())
        print("Batch label:", batch[0]["time_series"])
        print("Batch answer:", batch[0]["answer"])
        break
