#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Test script for ECG-QA dataset implementation.

This script tests:
1. ECG-QA repository cloning
2. PTB-XL dataset downloading  
3. Dataset loading and basic functionality
4. Sample data access

Usage:
    python test_ecgqa.py
"""
import sys

def test_ecgqa_loader():
    """Test the ECG-QA loader functions."""
    print("Testing ECG-QA loader...")
    
    try:
        from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import (
            does_ecg_qa_exist, 
            does_ptbxl_exist,
            download_ecg_qa_if_not_exists,
            download_ptbxl_if_not_exists
        )
        
        print(f"ECG-QA exists: {does_ecg_qa_exist()}")
        print(f"PTB-XL exists: {does_ptbxl_exist()}")
        
        # Download if needed (this might take a while)
        print("Ensuring datasets are available...")
        download_ecg_qa_if_not_exists()
        download_ptbxl_if_not_exists()
        
        print(f"After download - ECG-QA exists: {does_ecg_qa_exist()}")
        print(f"After download - PTB-XL exists: {does_ptbxl_exist()}")
        
        return True
        
    except Exception as e:
        print(f"Error in loader test: {e}")
        return False

def test_ecgqa_dataset():
    """Test the ECG-QA dataset class."""
    print("\nTesting ECGQADataset...")
    
    try:
        from opentslm.time_series_datasets.ecg_qa.ECGQADataset import ECGQADataset
        
        # Try to create dataset instances with limited samples for faster testing
        print("Creating dataset instances (limited to 5 samples each for testing)...")
        dataset = ECGQADataset(split="train", EOS_TOKEN="", max_samples=5)
        dataset_val = ECGQADataset(split="validation", EOS_TOKEN="", max_samples=5)  
        dataset_test = ECGQADataset(split="test", EOS_TOKEN="", max_samples=5)
        
        print(f"Dataset sizes:")
        print(f"  Train: {len(dataset)} samples")
        print(f"  Validation: {len(dataset_val)} samples")
        print(f"  Test: {len(dataset_test)} samples")
        
        if len(dataset) > 0:
            print(f"\nExamining first training sample:")
            sample = dataset[0]
            
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Question: {sample.get('question', 'N/A')}")
            print(f"Answer: {sample['answer']}")
            print(f"Question type: {sample.get('question_type', 'N/A')}")
            print(f"ECG IDs: {sample.get('ecg_id', 'N/A')}")
            
            if 'time_series_text' in sample:
                print(f"Number of time series: {len(sample['time_series_text'])}")
                
                # Show first time series info and data
                if len(sample['time_series_text']) > 0:
                    first_ts = sample['time_series_text'][0]
                    if hasattr(first_ts, 'text'):
                        print(f"First time series label: {first_ts.text}")
                        print(f"First time series length: {len(first_ts.time_series)}")
                        
                        # Show the actual ECG data values
                        print(f"First 20 ECG data points: {first_ts.time_series[:20]}")
                        print(f"ECG data range: min={min(first_ts.time_series):.4f}, max={max(first_ts.time_series):.4f}")
                        
                        # Show statistics
                        import numpy as np
                        ecg_data = np.array(first_ts.time_series)
                        print(f"ECG statistics: mean={np.mean(ecg_data):.4f}, std={np.std(ecg_data):.4f}")
                        
                        # Show a few more leads if available
                        if len(sample['time_series_text']) > 1:
                            print(f"\nOther ECG leads available:")
                            for i, ts in enumerate(sample['time_series_text'][1:4], 1):  # Show up to 3 more leads
                                if hasattr(ts, 'text'):
                                    print(f"  Lead {i+1}: {ts.text}")
                                    print(f"    First 10 values: {ts.time_series[:10]}")
                                    ecg_data = np.array(ts.time_series)
                                    print(f"    Stats: mean={np.mean(ecg_data):.4f}, std={np.std(ecg_data):.4f}")
                    else:
                        print(f"Time series format issue - got {type(first_ts)}: {first_ts}")
                        
                        # If it's a string, try to parse the actual time series data
                        if 'time_series' in sample:
                            print(f"\nRaw time_series field type: {type(sample['time_series'])}")
                            if isinstance(sample['time_series'], list) and len(sample['time_series']) > 0:
                                print(f"First time series data (first 20 points): {sample['time_series'][:20]}")
                                print(f"Time series length: {len(sample['time_series'])}")
                                
                                import numpy as np
                                ts_data = np.array(sample['time_series'])
                                print(f"Time series stats: mean={np.mean(ts_data):.4f}, std={np.std(ts_data):.4f}")
            
            print(f"\nPre-prompt: {sample['pre_prompt']}...")
            print(f"Post-prompt: {sample['post_prompt']}...")
        
        return True
        
    except ImportError as e:
        print(f"Import error (likely missing wfdb): {e}")
        print("Please install wfdb: pip install wfdb")
        return False
    except Exception as e:
        print(f"Error in dataset test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("ECG-QA Dataset Test Suite")
    print("=" * 50)
    
    # Test 1: Loader functionality
    loader_success = test_ecgqa_loader()
    
    # Test 2: Dataset functionality (only if loader works)
    if loader_success:
        dataset_success = test_ecgqa_dataset()
    else:
        dataset_success = False
        print("Skipping dataset test due to loader failure")
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  Loader test: {'✓ PASS' if loader_success else '✗ FAIL'}")
    print(f"  Dataset test: {'✓ PASS' if dataset_success else '✗ FAIL'}")
    
    if loader_success and dataset_success:
        print("\n🎉 All tests passed! ECG-QA dataset is ready to use.")
        print("\nTo use the full dataset (without sample limits), create ECGQADataset without max_samples parameter:")
        print("  dataset = ECGQADataset(split='train', EOS_TOKEN='')")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        
    return loader_success and dataset_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 