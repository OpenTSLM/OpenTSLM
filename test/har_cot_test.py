#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Test script for the HAR CoT loader and dataset.
"""

import unittest

# Import and set up global logger with verbose mode
from opentslm.logger import get_logger, set_global_verbose

def pretty_print_label_distribution(dataset, name):
    """Pretty print label distribution for a dataset."""
    from opentslm.time_series_datasets.har_cot.har_cot_loader import get_label_distribution
    label_dist = get_label_distribution(dataset)
    total = len(dataset)
    print(f"\n{name} dataset:")
    print(f"  Total samples: {total}")
    print(f"  Label distribution:")
    for label, count in sorted(label_dist.items()):
        print(f"    {label:15s}: {count:5d} ({count/total*100:5.1f}%)")

class TestHARCoTLoader(unittest.TestCase):
    """
    Unit tests for the HAR CoT loader functions.
    """
    def setUp(self):
        # Set up global logger with verbose mode for detailed output
        set_global_verbose(True)
        self.logger = get_logger()
        
        from opentslm.time_series_datasets.har_cot.har_cot_loader import load_har_cot_splits
        self.load_har_cot_splits = load_har_cot_splits
        
        self.logger.loading("Loading HAR CoT dataset splits...")
        self.train, self.val, self.test = self.load_har_cot_splits()
        self.logger.success(f"Dataset loaded successfully: Train={len(self.train)}, Val={len(self.val)}, Test={len(self.test)}")

    def test_dataset_sizes(self):
        """Test that the datasets are non-empty and splits are correct."""
        self.logger.info("Testing dataset sizes...")
        self.assertGreater(len(self.train), 0)
        self.assertGreater(len(self.val), 0)
        self.assertGreater(len(self.test), 0)
        self.logger.success("Dataset size tests passed")

    def test_label_distribution(self):
        """Test that label distributions are reasonable across splits."""
        self.logger.info("Testing label distributions...")
        pretty_print_label_distribution(self.train, "Train")
        pretty_print_label_distribution(self.val, "Validation")
        pretty_print_label_distribution(self.test, "Test")
        
        from opentslm.time_series_datasets.har_cot.har_cot_loader import get_label_distribution
        train_dist = get_label_distribution(self.train)
        val_dist = get_label_distribution(self.val)
        test_dist = get_label_distribution(self.test)
        
        # Check that all splits have the same labels
        expected_labels = {"biking", "lying", "running", "sitting", "standing", "walking", "walking_down", "walking_up"}
        self.assertEqual(set(train_dist.keys()), expected_labels)
        self.assertEqual(set(val_dist.keys()), expected_labels)
        self.assertEqual(set(test_dist.keys()), expected_labels)
        
        # Check that all splits have at least one sample per label
        for label in expected_labels:
            self.assertGreater(train_dist[label], 0, f"No samples for {label} in train")
            self.assertGreater(val_dist[label], 0, f"No samples for {label} in validation")
            self.assertGreater(test_dist[label], 0, f"No samples for {label} in test")
        
        self.logger.success("Label distribution tests passed")

    def test_sample_keys(self):
        """Test that a sample contains all required keys."""
        self.logger.info("Testing sample keys...")
        sample = self.train[0]
        required_keys = {"x_axis", "y_axis", "z_axis", "label", "rationale"}
        self.assertTrue(required_keys.issubset(sample.keys()))
        self.logger.success("Sample keys test passed")

    def test_axis_content(self):
        """Test that the axis data are lists and non-empty."""
        self.logger.info("Testing axis content...")
        sample = self.train[0]
        for axis in ["x_axis", "y_axis", "z_axis"]:
            self.assertIsInstance(sample[axis], list)
            self.assertGreater(len(sample[axis]), 0)
            # Check that all values are numeric
            for val in sample[axis]:
                self.assertIsInstance(val, (int, float))
            self.logger.debug(f"{axis}: length={len(sample[axis])}")
        
        # Check that all axes have the same length
        x_len = len(sample["x_axis"])
        y_len = len(sample["y_axis"])
        z_len = len(sample["z_axis"])
        self.assertEqual(x_len, y_len, "x_axis and y_axis have different lengths")
        self.assertEqual(y_len, z_len, "y_axis and z_axis have different lengths")
        self.logger.success("Axis content tests passed")

    def test_label_is_string(self):
        """Test that the label is a string and valid."""
        self.logger.info("Testing label format...")
        sample = self.train[0]
        self.assertIsInstance(sample["label"], str)
        self.assertGreater(len(sample["label"]), 0)
        
        # Check that label is one of the expected values
        expected_labels = {"biking", "lying", "running", "sitting", "standing", "walking", "walking_down", "walking_up"}
        self.assertIn(sample["label"], expected_labels)
        self.logger.success(f"Label test passed: '{sample['label']}'")

    def test_rationale_content(self):
        """Test that the rationale is a string and non-empty."""
        self.logger.info("Testing rationale content...")
        sample = self.train[0]
        self.assertIsInstance(sample["rationale"], str)
        self.assertGreater(len(sample["rationale"]), 0)
        self.logger.success(f"Rationale test passed: length={len(sample['rationale'])}")

    def test_example_data(self):
        """Print example data to show what the dataset looks like."""
        sample = self.train[0]
        self.logger.info("="*80)
        self.logger.info("EXAMPLE HAR COT DATASET SAMPLE")
        self.logger.info("="*80)
        self.logger.info(f"Label: '{sample['label']}'")
        if 'rationale' in sample:
            rationale_preview = sample['rationale'][:200] + "..." if len(sample['rationale']) > 200 else sample['rationale']
            self.logger.info(f"Rationale: '{rationale_preview}'")
        for axis in ["x_axis", "y_axis", "z_axis"]:
            self.logger.info(f"{axis}: length={len(sample[axis])}, first 5: {sample[axis][:5]}, last 5: {sample[axis][-5:]}")
        self.logger.info("="*80)

class TestHARCoTQADataset(unittest.TestCase):
    """
    Unit tests for the HARCoTQADataset class.
    """
    def setUp(self):
        # Set up global logger with verbose mode for detailed output
        set_global_verbose(True)
        self.logger = get_logger()
        
        from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
        self.HARCoTQADataset = HARCoTQADataset
        
        self.logger.loading("Initializing HARCoTQADataset...")
        self.train_dataset = self.HARCoTQADataset(split="train", EOS_TOKEN="")
        self.val_dataset = self.HARCoTQADataset(split="validation", EOS_TOKEN="")
        self.test_dataset = self.HARCoTQADataset(split="test", EOS_TOKEN="")
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
        required_keys = {"answer", "pre_prompt", "post_prompt", "time_series", "time_series_text", "label", "x_axis", "y_axis", "z_axis"}
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
        
        # Should have 3 time series (x, y, z axes)
        self.assertEqual(len(sample["time_series"]), 3)
        self.assertEqual(len(sample["time_series_text"]), 3)
        
        # Each time series should be non-empty
        for i, ts in enumerate(sample["time_series"]):
            self.assertGreater(len(ts), 0, f"Time series {i} is empty")
            # Check that all values are numeric
            for val in ts:
                self.assertIsInstance(val, (int, float))
        
        # Each time series text should be a string
        for i, ts_text in enumerate(sample["time_series_text"]):
            self.assertIsInstance(ts_text, str, f"Time series text {i} is not a string")
            self.assertGreater(len(ts_text), 0, f"Time series text {i} is empty")
        
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

    def test_labels_static_method(self):
        """Test that the static get_labels method returns expected labels."""
        self.logger.info("Testing static get_labels method...")
        labels = self.HARCoTQADataset.get_labels()
        expected_labels = ["biking", "lying", "running", "sitting", "standing", "walking", "walking_down", "walking_up"]
        self.assertEqual(labels, expected_labels)
        self.logger.success(f"Static labels test passed: {labels}")

    def test_prompts_content(self):
        """Test that prompts contain expected content."""
        self.logger.info("Testing prompt content...")
        sample = self.train_dataset[0]
        
        # Pre-prompt should mention accelerometer and activities
        pre_prompt = sample["pre_prompt"]
        self.assertIn("accelerometer", pre_prompt.lower())
        self.assertIn("activity", pre_prompt.lower())
        
        # Should mention some of the activity labels
        activity_mentions = 0
        for label in self.HARCoTQADataset.get_labels():
            if label in pre_prompt:
                activity_mentions += 1
        self.assertGreater(activity_mentions, 0, "Pre-prompt doesn't mention any activity labels")
        
        # Post-prompt should be simple
        post_prompt = sample["post_prompt"]
        self.assertIsInstance(post_prompt, str)
        self.assertGreater(len(post_prompt), 0)
        
        self.logger.success("Prompt content tests passed")

    def test_example_data_QA(self):
        """Print example data for HARCoTQADataset, showing all time series and text."""
        sample = self.train_dataset[0]
        self.logger.info("="*80)
        self.logger.info("EXAMPLE HARCoTQADataset SAMPLE")
        self.logger.info("="*80)
        self.logger.info(f"Label: '{sample['label']}'")
        pre_prompt_preview = sample['pre_prompt'][:200] + "..." if len(sample['pre_prompt']) > 200 else sample['pre_prompt']
        self.logger.info(f"Pre-prompt: '{pre_prompt_preview}'")
        self.logger.info(f"Post-prompt: '{sample['post_prompt']}'")
        answer_preview = sample['answer'][:200] + "..." if len(sample['answer']) > 200 else sample['answer']
        self.logger.info(f"Answer (rationale): '{answer_preview}'")
        self.logger.info(f"Number of time series: {len(sample['time_series'])}")
        for i, (ts, ts_text) in enumerate(zip(sample['time_series'], sample['time_series_text'])):
            self.logger.info(f"Time series {i} text: '{ts_text}'")
            self.logger.info(f"Time series {i} length: {len(ts)}")
            self.logger.info(f"First 10 values: {ts[:10]}")
            self.logger.info(f"Last 10 values: {ts[-10:]}")
        self.logger.info("="*80)

if __name__ == "__main__":
    print("Running HAR CoT loader and dataset tests, this might take a while...")
    unittest.main() 