# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
M4QADataset.py
-------------
PyTorch-style QA dataset for M4 time series caption generation.

This module defines the M4QADataset class, which wraps M4 time series and captions
for use in question-answering and caption generation tasks.
"""
import json
from typing import List, Literal, Tuple
import torch
from datasets import Dataset

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.m4.m4_loader import load_all_m4_data, create_combined_dataset


class M4QADataset(QADataset):
    """
    M4 Question-Answer Dataset for time series caption generation.
    
    This dataset loads M4 time series data with corresponding ChatGPT-created captions and creates
    QA pairs where the question asks for a caption and the answer is the actual caption.
    The model learns to generate detailed captions from time series data.
    """
    
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load and split the M4 data into train/validation/test sets.
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Load all M4 data across frequencies
        data_dict = load_all_m4_data()
        # Create combined dataset with splits
        train, val, test = create_combined_dataset(data_dict, seed=42)
        return train, val, test

    def _get_answer(self, row) -> str:
        """
        Get the caption as the answer.
        Args:
            row: Dataset row containing caption data
        Returns:
            The caption string
        """
        return row['caption']

    def _get_pre_prompt(self, row) -> str:
        """
        Get the pre-prompt asking to check the data.
        Args:
            row: Dataset row
        Returns:
            Question string asking to examine the time series
        """
        return "You are an expert in time series analysis."

    def _get_post_prompt(self, row) -> str:
        """
        Get the post-prompt asking for a detailed caption.
        Args:
            row: Dataset row
        Returns:
            Post-prompt string asking for caption generation
        """
        return "Please generate a detailed caption for this time-series, describing it as accurately as possible."

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Create text-time series prompts from the data.
        Args:
            row: Dataset row containing series data
        Returns:
            List of TextTimeSeriesPrompt objects with the time series
        """
        series = row['series']
        # Convert series to tensor and normalize
        if isinstance(series, list):
            series_tensor = torch.tensor(series, dtype=torch.float32)
        else:
            series_tensor = series
        # Normalize the series
        mean = series_tensor.mean()
        std = series_tensor.std()
        if std > 0:
            normalized_series = (series_tensor - mean) / std
        else:
            normalized_series = series_tensor - mean
        # Create the prompt with mean and std information
        text_prompt = f"This is the time series, it has mean {mean:.4f} and std {std:.4f}:"
        return [TextTimeSeriesPrompt(text_prompt, normalized_series.tolist())]

    def _format_sample(self, row):
        """Override to preserve the time series ID."""
        # Get the base formatted sample
        base_sample = super()._format_sample(row)
        
        # Add the ID if it exists in the original row
        if 'id' in row:
            base_sample['id'] = row['id']
        
        return base_sample


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Create dataset instances
    train_dataset = M4QADataset("train", "")
    val_dataset = M4QADataset("validation", "")
    test_dataset = M4QADataset("test", "")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    # Example sample
    sample = train_dataset[0]
    print(f"\nExample sample:")
    print(f"Pre-prompt: {sample['pre_prompt']}")
    print(f"Post-prompt: {sample['post_prompt']}")
    print(f"Answer (caption): {sample['answer'][:200]}...")
    print(f"Number of time series prompts: {len(sample['time_series'])}")
    if sample['time_series']:
        ts = sample['time_series'][0]
        ts_text = sample['time_series_text'][0]
        print(f"Time series length: {len(ts)}")
        print(f"Time series text: {ts_text}") 