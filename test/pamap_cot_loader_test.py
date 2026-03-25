#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Test script for the PAMAP2 CoT loader.
"""

import unittest
from opentslm.logger import get_logger, set_global_verbose
from opentslm.time_series_datasets.pamap2.BalancedBatchSampler import BalancedBatchSampler

# Import and set up global logger with verbose mode
from opentslm.logger import get_logger, set_global_verbose
from opentslm.time_series_datasets.pamap2.BalancedBatchSampler import BalancedBatchSampler
class TestPAMAP2CoTLoader(unittest.TestCase):
    """
    Unit tests for the PAMAP2 CoT loader functions.
    """
    def setUp(self):
        # Set up global logger with verbose mode for detailed output
        set_global_verbose(True)
        self.logger = get_logger()
        
        from opentslm.time_series_datasets.pamap2.pamap2_cot_loader import load_pamap2_cot_splits
        self.load_pamap2_cot_splits = load_pamap2_cot_splits
        
        self.logger.loading("Loading PAMAP2 CoT dataset splits...")
        self.train, self.val, self.test = self.load_pamap2_cot_splits()
        self.logger.success(f"Dataset loaded successfully: Train={len(self.train)}, Val={len(self.val)}, Test={len(self.test)}")

    def test_dataset_sizes(self):
        """Test that the datasets are non-empty and splits are correct."""
        self.logger.info("Testing dataset sizes...")
        self.assertGreater(len(self.train), 0)
        self.assertGreater(len(self.val), 0)
        self.assertGreater(len(self.test), 0)
        self.logger.success("Dataset size tests passed")

    def test_sample_keys(self):
        """Test that a sample contains all required keys."""
        self.logger.info("Testing sample keys...")
        sample = self.train[0]
        required_keys = {"x_axis", "y_axis", "z_axis", "label"}
        self.assertTrue(required_keys.issubset(sample.keys()))
        self.logger.success("Sample keys test passed")

    def test_axis_content(self):
        """Test that the axis data are lists and non-empty."""
        self.logger.info("Testing axis content...")
        sample = self.train[0]
        for axis in ["x_axis", "y_axis", "z_axis"]:
            self.assertIsInstance(sample[axis], list)
            self.assertGreater(len(sample[axis]), 0)
            self.logger.debug(f"{axis}: length={len(sample[axis])}")
        self.logger.success("Axis content tests passed")

    def test_label_is_string(self):
        """Test that the label is a string and non-empty."""
        self.logger.info("Testing label format...")
        sample = self.train[0]
        self.assertIsInstance(sample["label"], str)
        self.assertGreater(len(sample["label"]), 0)
        self.logger.success(f"Label test passed: '{sample['label']}'")

    def test_example_data(self):
        """Print example data to show what the dataset looks like."""
        sample = self.train[0]
        self.logger.info("="*80)
        self.logger.info("EXAMPLE PAMAP2 COT DATASET SAMPLE")
        self.logger.info("="*80)
        self.logger.info(f"Label: '{sample['label']}'")
        if 'rationale' in sample:
            self.logger.info(f"Rationale: '{sample['rationale']}'")
        for axis in ["x_axis", "y_axis", "z_axis"]:
            self.logger.info(f"{axis}: length={len(sample[axis])}, first 5: {sample[axis][:5]}")
        self.logger.info("="*80)

class TestPAMAP2CoTQADataset(unittest.TestCase):
    """
    Unit tests for the PAMAP2CoTQADataset class.
    """
    def setUp(self):
        # Set up global logger with verbose mode for detailed output
        set_global_verbose(True)
        self.logger = get_logger()
        
        from opentslm.time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
        self.PAMAP2CoTQADataset = PAMAP2CoTQADataset
        
        self.logger.loading("Initializing PAMAP2CoTQADataset...")
        self.train_dataset = self.PAMAP2CoTQADataset(split="train", EOS_TOKEN="")
        self.val_dataset = self.PAMAP2CoTQADataset(split="validation", EOS_TOKEN="")
        self.test_dataset = self.PAMAP2CoTQADataset(split="test", EOS_TOKEN="")
        self.logger.success(f"Datasets initialized: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")

    def test_dataset_sizes(self):
        """Test that the datasets are non-empty and splits are correct."""
        self.logger.info("Testing QA dataset sizes...")
        self.assertGreater(len(self.train_dataset), 0)
        self.assertGreater(len(self.val_dataset), 0)
        self.assertGreater(len(self.test_dataset), 0)
        self.logger.success("QA dataset size tests passed")

    def test_sample_keys(self):
        """Test that a sample contains all required keys."""
        self.logger.info("Testing QA sample keys...")
        sample = self.train_dataset[0]
        required_keys = {"answer", "pre_prompt", "post_prompt", "time_series", "time_series_text"}
        self.assertTrue(required_keys.issubset(sample.keys()))
        self.logger.success("QA sample keys test passed")

    def test_answer_is_rationale(self):
        """Test that the answer is a string (rationale)."""
        self.logger.info("Testing answer format...")
        sample = self.train_dataset[0]
        self.assertIsInstance(sample["answer"], str)
        self.assertGreater(len(sample["answer"]), 0)
        self.logger.success(f"Answer test passed: length={len(sample['answer'])}")

    def test_time_series_content(self):
        """Test that the time series and text are present and valid."""
        self.logger.info("Testing time series content...")
        sample = self.train_dataset[0]
        self.assertIsInstance(sample["time_series"], list)
        self.assertIsInstance(sample["time_series_text"], list)
        self.assertGreater(len(sample["time_series"][0]), 0)
        self.assertIsInstance(sample["time_series_text"][0], str)
        self.logger.success(f"Time series test passed: {len(sample['time_series'])} series")

    def test_time_series_text_includes_mean_std(self):
        """Test that each time_series_text includes 'mean' and 'std', and both are followed by a number."""
        import re
        self.logger.info("Testing time series text format...")
        sample = self.train_dataset[0]
        for i, text in enumerate(sample['time_series_text']):
            self.logger.debug(f"Testing text {i}: {text[:100]}...")
            self.assertIn('mean', text)
            self.assertIn('std', text)
            # Allow for any whitespace after 'mean' and 'std'
            mean_match = re.search(r"mean\s+(-?\d+\.\d+)", text)
            if not mean_match:
                self.logger.error(f"DEBUG: {repr(text)}")
            self.assertIsNotNone(mean_match, f"No mean value found in: {repr(text)}")
            std_match = re.search(r"std\s+(-?\d+\.\d+)", text)
            if not std_match:
                self.logger.error(f"DEBUG: {repr(text)}")
            self.assertIsNotNone(std_match, f"No std value found in: {repr(text)}")
        self.logger.success("Time series text format tests passed")

    def test_example_data_QA(self):
        """Print example data for PAMAP2CoTQADataset, showing all time series and text."""
        sample = self.train_dataset[0]
        self.logger.info("="*80)
        self.logger.info("EXAMPLE PAMAP2CoTQADataset SAMPLE")
        self.logger.info("="*80)
        self.logger.info(f"Pre-prompt: '{sample['pre_prompt']}'")
        self.logger.info(f"Post-prompt: '{sample['post_prompt']}'")
        self.logger.info(f"Answer (rationale): '{sample['answer']}'")
        self.logger.info(f"Number of time series: {len(sample['time_series'])}")
        for i, (ts, ts_text) in enumerate(zip(sample['time_series'], sample['time_series_text'])):
            self.logger.info(f"Time series {i} text: '{ts_text}'")
            self.logger.info(f"Time series {i} length: {len(ts)}")
            self.logger.info(f"First 10 values: {ts[:10]}")
            self.logger.info(f"Last 10 values: {ts[-10:]}")
        self.logger.info("="*80)

class TestBalancedBatchSampler(unittest.TestCase):
    def test_balanced_batches(self):
        """
        Test that BalancedBatchSampler produces batches where each class is equally represented.
        This test creates an imbalanced label list (10 'a', 4 'b', 6 'c') and sets batch_size=6 (2 samples per class per batch).
        For each batch yielded by the sampler, we check that every class appears exactly 2 times.
        This ensures the sampler yields perfectly balanced mini-batches, even when the dataset is imbalanced.
        """
        # Create a toy label list with imbalance
        labels = ['a'] * 10 + ['b'] * 4 + ['c'] * 6
        batch_size = 6  # 3 classes, so 2 samples per class per batch
        sampler = BalancedBatchSampler(labels, batch_size)
        for batch in sampler:
            batch_labels = [labels[idx] for idx in batch]
            counts = {l: batch_labels.count(l) for l in set(batch_labels)}
            print(f"Batch labels: {batch_labels}")
            print(f"Class counts in batch: {counts}")
            # Each class should appear exactly 2 times per batch
            for count in counts.values():
                self.assertEqual(count, 2, f"Expected 2 samples per class per batch, got {count}")

    def test_balanced_batches_pamap2cot(self):
        """
        Test that BalancedBatchSampler produces balanced batches on the real PAMAP2CoTQADataset training split.
        Prints batch labels and class counts for each batch.
        """
        from opentslm.time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
        from opentslm.time_series_datasets.pamap2.BalancedBatchSampler import BalancedBatchSampler
        # Helper to extract label from answer string
        def extract_label_from_answer(answer: str) -> str:
            # Assumes answer ends with 'Answer: <label>' or 'Answer: <label>.'
            if 'Answer:' in answer:
                label = answer.split('Answer:')[-1].strip()
                # Remove trailing period if present
                if label.endswith('.'):
                    label = label[:-1]
                return label.strip()
            return ''
        # Load the real dataset
        dataset = PAMAP2CoTQADataset(split="train", EOS_TOKEN="")
        labels = [extract_label_from_answer(row["answer"]) for row in dataset]
        num_classes = len(set(labels))
        batch_size = num_classes * 2  # 2 samples per class per batch
        sampler = BalancedBatchSampler(labels, batch_size)
        for i, batch in enumerate(sampler):
            batch_labels = [labels[idx] for idx in batch]
            counts = {l: batch_labels.count(l) for l in set(batch_labels)}
            print(f"Batch {i} labels: {batch_labels}")
            print(f"Batch {i} class counts: {counts}")
            for l, count in counts.items():
                self.assertEqual(count, 2, f"Expected 2 samples for class {l} per batch, got {count}")

if __name__ == "__main__":
    unittest.main() 