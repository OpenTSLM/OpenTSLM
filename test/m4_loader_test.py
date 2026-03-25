#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Test script for the M4QADataset with caption generation.
"""

import unittest
from opentslm.time_series_datasets.m4.M4QADataset import M4QADataset as _M4QADataset

class TestM4QADataset(unittest.TestCase):
    """
    Unit tests for the M4QADataset class and loader.
    """
    def setUp(self):
        self.M4QADataset = _M4QADataset
        self.train_dataset = self.M4QADataset("train", "")
        self.val_dataset = self.M4QADataset("validation", "")
        self.test_dataset = self.M4QADataset("test", "")

    def test_dataset_sizes(self):
        """Test that the datasets are non-empty and splits are correct."""
        self.assertGreater(len(self.train_dataset), 0)
        self.assertGreater(len(self.val_dataset), 0)
        self.assertGreater(len(self.test_dataset), 0)

    def test_sample_keys(self):
        """Test that a sample contains all required keys."""
        sample = self.train_dataset[0]
        required_keys = {"answer", "post_prompt", "pre_prompt", "time_series", "time_series_text"}
        self.assertTrue(required_keys.issubset(sample.keys()))

    def test_time_series_content(self):
        """Test that the time series and text are present and valid."""
        sample = self.train_dataset[0]
        self.assertIsInstance(sample["time_series"], list)
        self.assertIsInstance(sample["time_series_text"], list)
        self.assertGreater(len(sample["time_series"][0]), 0)
        self.assertIsInstance(sample["time_series_text"][0], str)

    def test_caption_is_answer(self):
        """Test that the answer is a string (caption)."""
        sample = self.train_dataset[0]
        self.assertIsInstance(sample["answer"], str)
        self.assertGreater(len(sample["answer"]), 0)

    def test_example_data(self):
        """Print example data to show what the dataset looks like."""
        sample = self.train_dataset[0]
        print("\n" + "="*80)
        print("EXAMPLE M4 DATASET SAMPLE")
        print("="*80)
        print(f"Pre-prompt: '{sample['pre_prompt']}'")
        print(f"Post-prompt: '{sample['post_prompt']}'")
        print(f"Answer (caption): '{sample['answer'][:200]}...'")
        print(f"Number of time series: {len(sample['time_series'])}")
        if sample['time_series']:
            ts = sample['time_series'][0]
            ts_text = sample['time_series_text'][0]
            print(f"Time series text: '{ts_text}'")
            print(f"Time series length: {len(ts)}")
            print(f"First 10 time series values: {ts[:10]}")
            print(f"Last 10 time series values: {ts[-10:]}")
        print("="*80)

if __name__ == "__main__":
    unittest.main()
