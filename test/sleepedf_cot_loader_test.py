#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Test script for the SleepEDF CoT loader.
"""

import unittest
from opentslm.time_series_datasets.sleep.sleepedf_cot_loader import load_sleepedf_cot_splits, get_label_distribution

def pretty_print_label_distribution(dataset, name):
    label_dist = get_label_distribution(dataset)
    total = len(dataset)
    print(f"\n{name} dataset:")
    print(f"  Total samples: {total}")
    print(f"  Label distribution:")
    for label, count in sorted(label_dist.items()):
        print(f"    {label:10s}: {count:5d} ({count/total*100:5.1f}%)")

class TestSleepEDFCotLoader(unittest.TestCase):
    """
    Unit tests for the SleepEDF CoT loader functions.
    """
    def setUp(self):
        self.train, self.val, self.test = load_sleepedf_cot_splits()

    def test_dataset_sizes(self):
        """Test that the datasets are non-empty and splits are correct."""
        self.assertGreater(len(self.train), 0)
        self.assertGreater(len(self.val), 0)
        self.assertGreater(len(self.test), 0)

    def test_label_stratification(self):
        """Test that label distributions are similar across splits."""
        pretty_print_label_distribution(self.train, "Train")
        pretty_print_label_distribution(self.val, "Validation")
        pretty_print_label_distribution(self.test, "Test")
        train_dist = get_label_distribution(self.train)
        val_dist = get_label_distribution(self.val)
        test_dist = get_label_distribution(self.test)
        # Check that all splits have at least one sample per label
        for label in train_dist:
            self.assertIn(label, val_dist)
            self.assertIn(label, test_dist)
            self.assertGreater(train_dist[label], 0)
            self.assertGreater(val_dist[label], 0)
            self.assertGreater(test_dist[label], 0)

    def test_sample_keys(self):
        """Test that a sample contains all required keys."""
        sample = self.train[0]
        required_keys = {"time_series", "label", "rationale"}
        self.assertTrue(required_keys.issubset(sample.keys()))

    def test_time_series_content(self):
        """Test that the time series is a list and non-empty."""
        sample = self.train[0]
        self.assertIsInstance(sample["time_series"], list)
        self.assertGreater(len(sample["time_series"]), 0)

    def test_label_is_string(self):
        """Test that the label is a string and non-empty."""
        sample = self.train[0]
        self.assertIsInstance(sample["label"], str)
        self.assertGreater(len(sample["label"]), 0)

    def test_example_data(self):
        """Print example data to show what the dataset looks like."""
        sample = self.train[0]
        print("="*80)
        print("EXAMPLE SLEEPEDF COT DATASET SAMPLE")
        print("="*80)
        print(f"Label: '{sample['label']}'")
        if 'rationale' in sample:
            print(f"Rationale: '{sample['rationale']}'")
        print(f"time_series: length={len(sample['time_series'])}, first 5: {sample['time_series'][:5]}")
        print("="*80)

if __name__ == "__main__":
    print("Running SleepEDF CoT loader tests, this might take a while...")
    unittest.main() 