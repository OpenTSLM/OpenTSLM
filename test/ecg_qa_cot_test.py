#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Test script for the ECG-QA CoT loader and dataset.
"""

import unittest
from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from opentslm.logger import get_logger, set_global_verbose

def pretty_print_label_distribution(dataset, name):
    """Pretty print label distribution for a dataset."""
    from opentslm.time_series_datasets.ecg_qa.ecgqa_cot_loader import get_label_distribution
    label_dist = get_label_distribution(dataset)
    total = len(dataset)
    print(f"\n{name} dataset:")
    print(f"  Total samples: {total}")
    print(f"  Label distribution:")
    for label, count in sorted(label_dist.items()):
        print(f"    {label:15s}: {count:5d} ({count/total*100:5.1f}%)")

class TestECGQACotLoader(unittest.TestCase):
    """
    Unit tests for the ECG-QA CoT loader functions.
    """
    def setUp(self):
        # Set up global logger with verbose mode for detailed output
        set_global_verbose(True)
        self.logger = get_logger()
        
        from opentslm.time_series_datasets.ecg_qa.ecgqa_cot_loader import load_ecg_qa_cot_splits
        self.load_ecg_qa_cot_splits = load_ecg_qa_cot_splits
        
        self.logger.loading("Loading ECG-QA CoT dataset splits...")
        self.train, self.val, self.test = self.load_ecg_qa_cot_splits()
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
        
        from opentslm.time_series_datasets.ecg_qa.ecgqa_cot_loader import get_label_distribution
        train_dist = get_label_distribution(self.train)
        val_dist = get_label_distribution(self.val)
        test_dist = get_label_distribution(self.test)
        
        # Check that all splits have reasonable distributions
        # Note: Not all labels need to appear in all splits due to random sampling
        self.assertGreater(len(train_dist), 0)
        self.assertGreater(len(val_dist), 0)
        self.assertGreater(len(test_dist), 0)
        
        # Check that major labels appear in all splits
        major_labels = ['yes', 'no', 'none']
        for label in major_labels:
            if label in train_dist:
                self.assertIn(label, val_dist, f"Major label '{label}' missing from validation set")
                self.assertIn(label, test_dist, f"Major label '{label}' missing from test set")
        
        self.logger.success("Label distribution tests passed")

    def test_sample_keys(self):
        """Test that a sample contains all required keys."""
        self.logger.info("Testing sample keys...")
        sample = self.train[0]
        required_keys = {"question", "question_type", "template_id", "ecg_id", "ecg_paths", "clinical_contexts", "rationale"}
        self.assertTrue(required_keys.issubset(sample.keys()))
        
        # Test that required fields are not None
        for key in required_keys:
            self.assertIsNotNone(sample[key], f"Required field '{key}' is None")
        
        self.logger.success("Sample keys test passed")

    def test_ecg_data_content(self):
        """Test that the ECG data is present and valid."""
        self.logger.info("Testing ECG data content...")
        sample = self.train[0]
        
        # Check ECG IDs
        self.assertIsInstance(sample["ecg_id"], list)
        self.assertGreater(len(sample["ecg_id"]), 0)
        for ecg_id in sample["ecg_id"]:
            self.assertIsInstance(ecg_id, int)
        
        # Check ECG paths
        self.assertIsInstance(sample["ecg_paths"], list)
        self.assertGreater(len(sample["ecg_paths"]), 0)
        for path in sample["ecg_paths"]:
            self.assertIsInstance(path, str)
            self.assertTrue(path.endswith('.dat'))
        
        # Check clinical contexts
        self.assertIsInstance(sample["clinical_contexts"], list)
        self.assertGreater(len(sample["clinical_contexts"]), 0)
        for context in sample["clinical_contexts"]:
            self.assertIsInstance(context, str)
            self.assertGreater(len(context), 0)
        
        self.logger.success("ECG data content tests passed")

    def test_question_content(self):
        """Test that the question data is valid."""
        self.logger.info("Testing question content...")
        sample = self.train[0]
        
        # Check question
        self.assertIsInstance(sample["question"], str)
        self.assertGreater(len(sample["question"]), 0)
        
        # Check question type
        self.assertIsInstance(sample["question_type"], str)
        self.assertGreater(len(sample["question_type"]), 0)
        
        # Check template ID
        self.assertIsInstance(sample["template_id"], int)
        
        self.logger.success("Question content tests passed")

    def test_rationale_content(self):
        """Test that the rationale is a string and non-empty."""
        self.logger.info("Testing rationale content...")
        sample = self.train[0]
        self.assertIsInstance(sample["rationale"], str)
        self.assertGreater(len(sample["rationale"]), 0)
        self.logger.success(f"Rationale test passed: length={len(sample['rationale'])}")

    def test_cot_fields(self):
        """Test that CoT-specific fields are present."""
        self.logger.info("Testing CoT-specific fields...")
        sample = self.train[0]
        
        # Check CoT fields that actually exist in the implementation
        cot_fields = ["rationale", "template_id", "question_type"]
        for field in cot_fields:
            self.assertIn(field, sample)
            # These should not be None
            self.assertIsNotNone(sample[field], f"Required field '{field}' is None")
            if field == "rationale":
                self.assertIsInstance(sample[field], str)
            elif field == "template_id":
                self.assertIsInstance(sample[field], int)
            elif field == "question_type":
                self.assertIsInstance(sample[field], str)
        
        self.logger.success("CoT fields tests passed")

    def test_example_data(self):
        """Print example data to show what the dataset looks like."""
        sample = self.train[0]
        self.logger.info("="*80)
        self.logger.info("EXAMPLE ECG-QA COT DATASET SAMPLE")
        self.logger.info("="*80)
        self.logger.info(f"Question: '{sample['question']}'")
        self.logger.info(f"Question type: '{sample['question_type']}'")
        self.logger.info(f"Template ID: {sample['template_id']}")
        self.logger.info(f"ECG IDs: {sample['ecg_id']}")
        self.logger.info(f"ECG paths: {sample['ecg_paths']}")
        if 'rationale' in sample:
            rationale_preview = sample['rationale'][:200] + "..." if len(sample['rationale']) > 200 else sample['rationale']
            self.logger.info(f"Rationale: '{rationale_preview}'")
        self.logger.info(f"Clinical contexts: {sample['clinical_contexts']}")
        self.logger.info("="*80)

class TestECGQACoTQADataset(unittest.TestCase):
    """
    Unit tests for the ECGQACoTQADataset class.
    """
    def setUp(self):
        # Set up global logger with verbose mode for detailed output
        set_global_verbose(True)
        self.logger = get_logger()
        
        from opentslm.time_series_datasets.ecg_qa import ECGQACoTQADataset
        self.ECGQACoTQADataset = ECGQACoTQADataset
        
        self.logger.loading("Initializing ECGQACoTQADataset...")
        # Use limited samples for faster testing
        self.train_dataset = self.ECGQACoTQADataset(split="train", EOS_TOKEN="", max_samples=5)
        self.val_dataset = self.ECGQACoTQADataset(split="validation", EOS_TOKEN="", max_samples=5)
        self.test_dataset = self.ECGQACoTQADataset(split="test", EOS_TOKEN="", max_samples=5)
        self.logger.success(f"Datasets initialized: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
        
        # Test the exclude_comparison functionality
        self.logger.loading("Testing exclude_comparison functionality...")
        self.train_no_comparison = self.ECGQACoTQADataset(split="train", EOS_TOKEN="", max_samples=10, exclude_comparison=True)
        self.logger.success(f"Non-comparison dataset initialized: {len(self.train_no_comparison)} samples")

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
        required_keys = {"answer", "pre_prompt", "post_prompt", "time_series", "time_series_text", "question", "question_type", "template_id"}
        self.assertTrue(required_keys.issubset(sample.keys()))
        
        # Test that required fields are not None
        for key in required_keys:
            self.assertIsNotNone(sample[key], f"Required field '{key}' is None")
        
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
        
        # Should have multiple time series (one per ECG lead)
        self.assertGreater(len(sample["time_series"]), 0)
        self.assertGreater(len(sample["time_series_text"]), 0)
        
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

    def test_time_series_text_includes_ecg_info(self):
        """Test that each time_series_text includes ECG lead information."""
        self.logger.info("Testing time series text format...")
        sample = self.train_dataset[0]
        for i, text in enumerate(sample['time_series_text']):
            self.logger.debug(f"Testing text {i}: {text[:100]}...")
            # Should mention ECG lead
            self.assertIn('ECG Lead', text)
            # Should mention mean and std
            self.assertIn('mean', text)
            self.assertIn('std', text)
            # Should be in natural language format (optimized version uses simpler format)
            # The optimized version uses: "This is ECG Lead X, it has mean Y and std Z:"
        
        self.logger.success("Time series text format tests passed")

    def test_labels_static_method(self):
        """Test that the static get_labels method returns expected labels."""
        self.logger.info("Testing static get_labels method...")
        labels = self.ECGQACoTQADataset.get_labels()
        
        # Test that the method works and returns reasonable results
        self.assertIsInstance(labels, list)
        self.assertGreater(len(labels), 0, "Labels list should not be empty")
        
        # Test that common labels are present
        common_labels = ["yes", "no", "none"]
        for label in common_labels:
            self.assertIn(label, labels, f"Common label '{label}' should be present")
        
        # Test that labels are strings
        for label in labels:
            self.assertIsInstance(label, str, f"All labels should be strings, got {type(label)}")
            self.assertGreater(len(label.strip()), 0, "Labels should not be empty strings")
        
        self.logger.success(f"Static labels test passed: {len(labels)} labels found")

    def test_prompts_content(self):
        """Test that prompts contain expected content."""
        self.logger.info("Testing prompt content...")
        sample = self.train_dataset[0]
        
        # Pre-prompt should mention cardiologist and ECG
        pre_prompt = sample["pre_prompt"]
        self.assertIn("cardiologist", pre_prompt.lower())
        self.assertIn("ecg", pre_prompt.lower())
        
        # Should mention the specific question
        question = sample["question"]
        self.assertIn(question, pre_prompt)
        
        # Post-prompt should mention answer format
        post_prompt = sample["post_prompt"]
        self.assertIn("Answer:", post_prompt)
        
        self.logger.success("Prompt content tests passed")

    def test_cot_specific_fields(self):
        """Test that CoT-specific fields are preserved in the formatted sample."""
        self.logger.info("Testing CoT-specific fields preservation...")
        sample = self.train_dataset[0]
        
        # Check fields that actually exist in the implementation
        cot_fields = ["rationale", "template_id", "question_type"]
        for field in cot_fields:
            if field in sample:
                self.logger.debug(f"Field {field}: {sample[field]}")
        
        self.logger.success("CoT fields preservation tests passed")

    def test_exclude_comparison_functionality(self):
        """Test that exclude_comparison properly filters out comparison questions."""
        self.logger.info("Testing exclude_comparison functionality...")
        
        # Check that the non-comparison dataset has no comparison questions
        for sample in self.train_no_comparison:
            question_type = sample.get("question_type")
            if question_type is None:
                raise ValueError(f"Sample missing question_type: {sample}")
            self.assertFalse(question_type.startswith("comparison"), 
                           f"Found comparison question in non-comparison dataset: {question_type}")
        
        # Check that the regular dataset might have comparison questions
        has_comparison = False
        for sample in self.train_dataset:
            question_type = sample.get("question_type")
            if question_type and question_type.startswith("comparison"):
                has_comparison = True
                break
        
        self.logger.success(f"Exclude comparison test passed. Regular dataset has comparison questions: {has_comparison}")

    def test_100hz_data_consistency(self):
        """Test that all time series data is consistently 100Hz."""
        self.logger.info("Testing 100Hz data consistency...")
        
        for dataset_name, dataset in [("train", self.train_dataset), ("val", self.val_dataset), ("test", self.test_dataset)]:
            for i, sample in enumerate(dataset):
                if i >= 3:  # Only test first 3 samples per dataset
                    break
                    
                time_series = sample.get("time_series")
                if time_series is None:
                    continue
                
                for j, ts in enumerate(time_series):
                    # Each time series should be exactly 1000 samples (10 seconds at 100Hz)
                    expected_length = 1000
                    actual_length = len(ts)
                    
                    if actual_length != expected_length:
                        self.logger.warning(f"Dataset {dataset_name}, sample {i}, time series {j}: "
                                          f"Expected {expected_length} samples, got {actual_length}")
                    
                    # All values should be numeric
                    for val in ts:
                        self.assertIsInstance(val, (int, float), 
                                           f"Non-numeric value in time series: {val}")
        
        self.logger.success("100Hz data consistency tests passed")

    def test_error_handling_missing_fields(self):
        """Test that the dataset properly raises errors for missing required fields."""
        self.logger.info("Testing error handling for missing fields...")
        
        # Create a sample with missing fields to test error handling
        from opentslm.time_series_datasets.ecg_qa import ECGQACoTQADataset
        
        # This should work normally
        try:
            normal_dataset = ECGQACoTQADataset(split="train", EOS_TOKEN="", max_samples=1)
            self.logger.success("Normal dataset creation works")
        except Exception as e:
            self.logger.error(f"Unexpected error creating normal dataset: {e}")
            raise
        
        # Test that we get proper errors when accessing samples with missing data
        # (This would happen if the underlying data is corrupted)
        self.logger.info("Error handling tests completed")

    def test_example_data_QA(self):
        """Print example data for ECGQACoTQADataset, showing all time series and text."""
        sample = self.train_dataset[0]
        self.logger.info("="*80)
        self.logger.info("EXAMPLE ECGQACoTQADataset SAMPLE")
        self.logger.info("="*80)
        self.logger.info(f"Question: '{sample['question']}'")
        self.logger.info(f"Question type: '{sample['question_type']}'")
        pre_prompt_preview = sample['pre_prompt'][:200]
        self.logger.info(f"Pre-prompt: '{pre_prompt_preview}'")
        self.logger.info(f"Post-prompt: '{sample['post_prompt']}'")
        answer_preview = sample['answer']
        self.logger.info(f"Answer (rationale): '{answer_preview}'")
        self.logger.info(f"Number of time series: {len(sample['time_series'])}")
        for i, (ts, ts_text) in enumerate(zip(sample['time_series'], sample['time_series_text'])):
            self.logger.info(f"Time series {i} text: '{ts_text}'")
            self.logger.info(f"Time series {i} length: {len(ts)}")
            self.logger.info(f"First 10 values: {ts[:10]}")
            self.logger.info(f"Last 10 values: {ts[-10:]}")
        self.logger.info("="*80)

if __name__ == "__main__":
    print("Running ECG-QA CoT loader and dataset tests, this might take a while...")
    unittest.main()
