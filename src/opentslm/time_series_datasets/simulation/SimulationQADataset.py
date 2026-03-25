# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from typing import List, Tuple
from datasets import Dataset

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset


DATASET_SIZE = 200


class SimulationQADataset(QADataset):
    def __init__(
        self,
        split,
        EOS_TOKEN,
        length: int = 100,
        num_series: int = 1,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        """
        Initialize SimulationQADataset with one or more time series of variable length.

        Args:
            split: Dataset split (train/test/validation) - all return the same single item
            EOS_TOKEN: End-of-sequence token
            length: Length of the generated time series (default: 100)
            num_series: Number of time series to generate (default: 1)
            format_sample_str: Whether to format as string
            time_series_format_function: Optional time series formatting function
        """
        self.length = length
        self.num_series = num_series

        super().__init__(
            split, EOS_TOKEN, format_sample_str, time_series_format_function
        )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Creates a dataset with 200 items, each with random time series data.
        Each item will have num_series time series of length elements.
        """
        all_items = []

        for _ in range(DATASET_SIZE):
            # Generate random time series for this item
            time_series_data = {}
            for i in range(self.num_series):
                series = torch.tensor(np.random.randn(self.length), dtype=torch.float32)

                # Normalize the series
                mean_val = series.mean().item()
                std_val = max(series.std().item(), 1e-6)
                normalized_series = (series - mean_val) / std_val

                time_series_data[f"series_{i}"] = normalized_series.tolist()
                time_series_data[f"series_text_{i}"] = (
                    f"This is a time series with mean {mean_val:.4f} and std {std_val:.4f}."
                )

            item_data = {
                **time_series_data,
                "Question": f"You are given different time series. All have the same length of {self.length} data points. What is the pattern of the time series?",
                "Answer": "This is a random pattern.",
            }

            all_items.append(item_data)

        # Convert to HuggingFace Dataset format
        dataset_dict = {}
        for key in all_items[0].keys():
            dataset_dict[key] = [item[key] for item in all_items]

        dataset = Dataset.from_dict(dataset_dict)

        # Return the same dataset for all splits
        return dataset, dataset, dataset

    def _get_answer(self, row) -> str:
        """Get the answer from the data row."""
        return row["Answer"]

    def _get_pre_prompt(self, row) -> str:
        """Get the question/pre-prompt from the data row."""
        return row["Question"]

    def _get_post_prompt(self, row) -> str:
        """Get the post-prompt from the data row."""
        return "Predict the pattern of the time series. Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the time series data from the current row to TextTimeSeriesPrompt format.
        Each row now contains its own random time series data.
        """
        prompts = []

        for i in range(self.num_series):
            series_key = f"series_{i}"
            series_text_key = f"series_text_{i}"
            if series_key in row and series_text_key in row:
                series_data = row[series_key]
                text_description = row[series_text_key]
                prompts.append(TextTimeSeriesPrompt(text_description, series_data))

        return prompts


if __name__ == "__main__":
    # Example usage - Single time series
    print("=== Single Time Series Dataset (10,000 items) ===")
    dataset_single = SimulationQADataset("train", "", length=50, num_series=1)
    print(f"Dataset length: {len(dataset_single)}")
    sample_single = dataset_single[0]
    print(f"Sample keys: {sample_single.keys()}")
    print(f"Question: {sample_single['pre_prompt'][:100]}...")
    print(f"Answer: {sample_single['answer']}")
    print(f"Number of time series prompts: {len(sample_single['time_series_prompts'])}")

    print("\n=== Multiple Time Series Dataset (10,000 items) ===")
    # Example usage - Multiple time series
    dataset_multi = SimulationQADataset("train", "", length=50, num_series=3)
    print(f"Dataset length: {len(dataset_multi)}")
    sample_multi = dataset_multi[0]
    print(f"Sample keys: {sample_multi.keys()}")
    print(f"Question: {sample_multi['pre_prompt'][:100]}...")
    print(f"Answer: {sample_multi['answer']}")
    print(f"Number of time series prompts: {len(sample_multi['time_series_prompts'])}")

    # Show time series prompt details
    for i, ts_prompt in enumerate(sample_multi["time_series_prompts"]):
        print(f"Time series {i}: {ts_prompt.text[:50]}...")

    # Test different splits (should all be the same)
    print("\n=== Testing Different Splits ===")
    train_dataset = SimulationQADataset("train", "", length=50, num_series=2)
    val_dataset = SimulationQADataset("validation", "", length=50, num_series=2)
    test_dataset = SimulationQADataset("test", "", length=50, num_series=2)

    print(f"Train length: {len(train_dataset)}")
    print(f"Val length: {len(val_dataset)}")
    print(f"Test length: {len(test_dataset)}")
    # Test that different items have different data
    print("\n=== Testing Randomness ===")
    item_0 = dataset_single[0]
    item_100 = dataset_single[100]
    print(f"Item 0 Series length: {len(item_0['time_series_prompts'][0].time_series)}")
    print(
        f"Item 100 Series length: {len(item_100['time_series_prompts'][0].time_series)}"
    )
    print(
        f"First 5 values of item 0: {item_0['time_series_prompts'][0].time_series[:5]}"
    )
    print(
        f"First 5 values of item 100: {item_100['time_series_prompts'][0].time_series[:5]}"
    )
    print(
        "Values are different:",
        item_0["time_series_prompts"][0].time_series[:5]
        != item_100["time_series_prompts"][0].time_series[:5],
    )