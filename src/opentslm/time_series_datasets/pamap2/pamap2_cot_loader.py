# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import os
import pandas as pd
from datasets import Dataset
from typing import Tuple, Dict
import ast
import urllib.request
import zipfile
import shutil
from opentslm.time_series_datasets.constants import RAW_DATA
import math
import logging
from opentslm.logger import get_logger


PAMAP_DATA_DIR = os.path.join(RAW_DATA, "pamap")
COT_CSV = os.path.join(PAMAP_DATA_DIR, "pamap2_cot.csv")
PAMAP_RELEASE_URL = "https://polybox.ethz.ch/index.php/s/mKJc8aX4nggScfs/download"


TEST_FRAC = 0.05
VAL_FRAC = 0.05


def download_and_extract_pamap2():
    """
    Download the PAMAP2 CoT CSV into data/pamap/ if not already present.
    """
    if os.path.exists(PAMAP_DATA_DIR) and os.path.exists(COT_CSV):
        print(f"PAMAP2 CoT dataset already exists at {COT_CSV}")
        return

    os.makedirs(PAMAP_DATA_DIR, exist_ok=True)
    print(f"Downloading PAMAP2 CoT dataset from {PAMAP_RELEASE_URL}...")
    try:
        urllib.request.urlretrieve(PAMAP_RELEASE_URL, COT_CSV)
        print("Download completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to download PAMAP2 CoT dataset: {e}")

    if not os.path.exists(COT_CSV):
        raise FileNotFoundError(f"pamap2_cot.csv not found after download in {PAMAP_DATA_DIR}")


def ensure_pamap2_cot_dataset():
    """
    Ensure the PAMAP2 CoT dataset is available in data/pamap/.
    Download and extract if necessary.
    """
    if not os.path.exists(COT_CSV):
        download_and_extract_pamap2()


def load_pamap2_cot_splits(seed: int = 42, min_series_length: int = 50) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the PAMAP2 CoT dataset and split it into train, validation, and test sets.
    Uses stratified splitting to ensure all classes are represented in each split.
    
    Args:
        seed: Random seed for reproducible splits
        min_series_length: Minimum length for time series (shorter series will be excluded)
        
    Returns:
        Tuple of (train, validation, test) datasets
    """
    logger = get_logger()
    
    ensure_pamap2_cot_dataset()
    
    if not os.path.exists(COT_CSV):
        raise FileNotFoundError(f"CoT CSV not found: {COT_CSV}")
        
    df = pd.read_csv(COT_CSV)
    
    def parse_series(s):
        try:
            series = ast.literal_eval(s)
            # Validate the parsed series
            if not isinstance(series, list):
                logger.error(f"Invalid series type: {type(series)}")
                logger.error(f"Raw data: {s[:200]}...")
                exit(1)
            # Check for NaN or infinite values
            for i, x in enumerate(series):
                if not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
                    logger.error(f"Invalid value detected at index {i}: {x} (type: {type(x)})")
                    logger.error(f"Full series: {series}")
                    logger.error(f"Raw data: {s[:200]}...")
                    exit(1)
            return series
        except (ValueError, SyntaxError) as e:
            logger.error(f"Failed to parse series: {e}")
            logger.error(f"Raw data: {s[:200]}...")
            exit(1)
    
    if 'x_axis' in df.columns:
        df['x_axis'] = df['x_axis'].apply(parse_series)
    if 'y_axis' in df.columns:
        df['y_axis'] = df['y_axis'].apply(parse_series)
    if 'z_axis' in df.columns:
        df['z_axis'] = df['z_axis'].apply(parse_series)
    
    # Filter out samples with series that are too short
    logger.info(f"Original dataset size: {len(df)}")
    
    # Check series lengths and filter
    valid_indices = []
    excluded_count = 0
    for idx, row in df.iterrows():
        x_len = len(row['x_axis']) if 'x_axis' in row else 0
        y_len = len(row['y_axis']) if 'y_axis' in row else 0
        z_len = len(row['z_axis']) if 'z_axis' in row else 0
        
        # Check if all series meet minimum length requirement
        if x_len >= min_series_length and y_len >= min_series_length and z_len >= min_series_length:
            valid_indices.append(idx)
        else:
            excluded_count += 1
            logger.debug(f"Excluding sample {idx}: x_len={x_len}, y_len={y_len}, z_len={z_len} (min required: {min_series_length})")
    
    df = df.iloc[valid_indices].reset_index(drop=True)
    logger.info(f"Filtered dataset size: {len(df)} (excluded {excluded_count} samples)")
    
    # Analyze class distribution before splitting
    logger.info("Class distribution before splitting:")
    class_counts = df['label'].value_counts()
    logger.info(f"Total classes: {len(class_counts)}")
    for label, count in class_counts.items():
        logger.debug(f"  {label}: {count} samples")
    
    # Check for classes with very few samples
    min_samples_per_class = 3  # Minimum samples needed per class for stratified splitting
    rare_classes = class_counts[class_counts < min_samples_per_class]
    if len(rare_classes) > 0:
        logger.warning(f"Classes with fewer than {min_samples_per_class} samples:")
        for label, count in rare_classes.items():
            logger.warning(f"  {label}: {count} samples")
        logger.warning("These classes may not be represented in all splits.")
    
    # Perform stratified splitting
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_FRAC, 
        random_state=seed, 
        stratify=df['label']
    )
    
    # Second split: train vs val
    val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_frac_adj, 
        random_state=seed+1, 
        stratify=train_val_df['label']
    )
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Print detailed split information
    logger.success("Pamap2CoT stratified split results:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Validation: {len(val_df)} samples") 
    logger.info(f"  Test: {len(test_df)} samples")
    
    # Print detailed per-class distribution for each split
    logger.info("Per-class sample distribution:")
    for split_name, split_df in [("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
        logger.info(f"{split_name} SET ({len(split_df)} total samples):")
        split_class_counts = split_df['label'].value_counts().sort_index()
        for label, count in split_class_counts.items():
            percentage = (count / len(split_df)) * 100
            logger.info(f"  {label:20s}: {count:3d} samples ({percentage:5.1f}%)")
    
    # Verify class representation in each split
    logger.info("Class representation verification:")
    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        split_classes = split_df['label'].value_counts()
        logger.info(f"  {split_name}: {len(split_classes)} classes represented")
        if len(split_classes) < len(class_counts):
            missing_classes = set(class_counts.index) - set(split_classes.index)
            logger.warning(f"    Missing classes: {missing_classes}")
    
    return train_dataset, val_dataset, test_dataset


def get_label_distribution(dataset: Dataset) -> Dict[str, int]:
    """
    Get the distribution of labels in a dataset.
    
    Args:
        dataset: The dataset to analyze.
    Returns:
        Dictionary mapping label to count.
    """
    labels = dataset['label']
    return dict(pd.Series(labels).value_counts())

def print_dataset_info(dataset: Dataset, name: str):
    """
    Print information about a dataset split.
    
    Args:
        dataset: The dataset split.
        name: Name of the split (e.g., 'Train').
    """
    label_dist = get_label_distribution(dataset)
    print(f"\n{name} dataset:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Label distribution:")
    for label, count in label_dist.items():
        print(f"    {label}: {count} ({count/len(dataset)*100:.1f}%)")


if __name__ == "__main__":
    print("=== PAMAP2CoT Dataset Loading Demo ===\n")
    
    # Demo with verbose mode
    print("1. Loading with verbose mode (detailed output):")
    train_ds, val_ds, test_ds = load_pamap2_cot_splits(verbose=True)
    
    print("\n" + "="*50 + "\n")
    
    # Demo without verbose mode
    print("2. Loading without verbose mode (minimal output):")
    train_ds, val_ds, test_ds = load_pamap2_cot_splits(verbose=False)
    
    print("\n" + "="*50 + "\n")
    
    # Show sample data
    if len(train_ds) > 0:
        print("3. Sample data from training set:")
        sample = train_ds[0]
        for key, value in sample.items():
            if key in ['x_axis', 'y_axis', 'z_axis']:
                print(f"{key}: {value[:5]}... (truncated)")
            else:
                print(f"{key}: {value}")
