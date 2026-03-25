# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
m4_loader.py
------------
Loader utilities for the M4 time series dataset with captions.

This module provides functions to load, merge, and split the processed M4 time series and caption data
for use in machine learning tasks such as time series caption generation.

Data source: https://github.com/StanfordBDHG/M4TimeSeriesCaptionDataset
Expected data location: data/M4TimeSeriesCaptionDataset/generated/
"""
import os
import json
import pandas as pd
import numpy as np
import subprocess
import shutil
import urllib.request
import zipfile
from typing import Dict, List, Literal, Optional, Tuple
from datasets import Dataset
from sklearn.model_selection import train_test_split
from opentslm.time_series_datasets.constants import RAW_DATA

# ---------------------------
# Constants
# ---------------------------

RELEASE_URL = "https://polybox.ethz.ch/index.php/s/MT3y9WdEebT8wfj/download/M4TimeSeriesCaptionDatasetV02.zip"


DATA_DIR = os.path.join(RAW_DATA, "M4TimeSeriesCaptionDataset")
GENERATED_DATA_DIR = os.path.join(DATA_DIR, "M4TimeSeriesCaptionDataset")

AVAILABLE_FREQUENCIES = ["Daily", "Hourly", "Monthly", "Quarterly", "Weekly", "Yearly"]

TEST_FRAC = 0.1
VAL_FRAC = 0.1

# ---------------------------
# Data download and setup
# ---------------------------

def download_and_extract_dataset():
    """
    Download the M4TimeSeriesCaptionDataset zip file and extract it.
    """
    if os.path.exists(GENERATED_DATA_DIR):
        print(f"Dataset already exists at {GENERATED_DATA_DIR}")
        return
     
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download the zip file
    zip_path = os.path.join(DATA_DIR, "M4TimeSeriesCaptionDatasetv01.zip")
    print(f"Downloading dataset from {RELEASE_URL}...")
    
    try:
        urllib.request.urlretrieve(RELEASE_URL, zip_path)
        print("Download completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")
    
    # Extract the zip file
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract dataset: {e}")
    
    # Clean up the zip file
    try:
        os.remove(zip_path)
        print("Cleaned up zip file.")
    except Exception as e:
        print(f"Warning: Could not remove zip file: {e}")
    
    if not os.path.exists(GENERATED_DATA_DIR):
        raise FileNotFoundError(f"Generated data directory not found after extraction: {GENERATED_DATA_DIR}")

def ensure_m4_dataset():
    """
    Ensure the M4TimeSeriesCaptionDataset is available.
    If not present, download and extract it from the GitHub release.
    """
    if not os.path.exists(GENERATED_DATA_DIR):
        download_and_extract_dataset()

def get_data_file_path(frequency: str, file_type: str) -> str:
    """
    Get the path to a data file in the dataset.
    
    Args:
        frequency: The frequency (Monthly, Quarterly, Weekly)
        file_type: Either 'series' or 'captions'
        
    Returns:
        Path to the data file
    """
    return os.path.join(GENERATED_DATA_DIR, f"m4_{file_type}_{frequency}.csv")

# ---------------------------
# Core loader
# ---------------------------

def load_m4_data(frequency: Literal["Monthly", "Quarterly", "Weekly"]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the M4 series and captions data for a given frequency.
    
    Args:
        frequency: One of ["Monthly", "Quarterly", "Weekly"]
        
    Returns:
        Tuple of (series_df, captions_df) where:
        - series_df has columns ["id", "series"] 
        - captions_df has columns ["id", "caption"]
    Raises:
        ValueError: If frequency is not supported or no common IDs are found.
        FileNotFoundError: If the required CSV files are missing.
    """
    if frequency not in AVAILABLE_FREQUENCIES:
        raise ValueError(f"Frequency must be one of {AVAILABLE_FREQUENCIES}")
    
    # Ensure dataset is available
    ensure_m4_dataset()
    
    # Load series data
    series_file = get_data_file_path(frequency, "series")
    if not os.path.exists(series_file):
        raise FileNotFoundError(f"Series file not found: {series_file}")
    
    series_df = pd.read_csv(series_file)
    
    # Load captions data
    captions_file = get_data_file_path(frequency, "captions")
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found: {captions_file}")
    
    captions_df = pd.read_csv(captions_file)
    
    # Ensure both dataframes have the same IDs
    series_ids = set(series_df['id'])
    caption_ids = set(captions_df['id'])
    common_ids = series_ids.intersection(caption_ids)
    
    if len(common_ids) == 0:
        raise ValueError(f"No common IDs found between series and captions for frequency {frequency}")
    
    # Filter to common IDs
    series_df = series_df[series_df['id'].isin(common_ids)].reset_index(drop=True)
    captions_df = captions_df[captions_df['id'].isin(common_ids)].reset_index(drop=True)
    
    print(f"Loaded {len(series_df)} samples for frequency {frequency}")
    
    return series_df, captions_df

def load_all_m4_data() -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load M4 data for all available frequencies.
    
    Returns:
        Dictionary mapping frequency to (series_df, captions_df) tuple
    """
    data = {}
    for frequency in AVAILABLE_FREQUENCIES:
        try:
            series_df, captions_df = load_m4_data(frequency)
            data[frequency] = (series_df, captions_df)
        except Exception as e:
            print(f"Warning: Could not load data for frequency {frequency}: {e}")
    
    return data

def create_combined_dataset(
    data_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train/val/test splits from combined data across all frequencies.
    
    Args:
        data_dict: Dictionary mapping frequency to (series_df, captions_df) tuple
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    all_samples = []
    
    for frequency, (series_df, captions_df) in data_dict.items():
        # Merge series and captions data
        merged_df = series_df.merge(captions_df, on='id', how='inner')
        
        # Convert to list of dictionaries
        for _, row in merged_df.iterrows():
            # Parse the series string to list of floats
            try:
                series_str = row['series']
                if isinstance(series_str, str):
                    series_list = json.loads(series_str)
                else:
                    series_list = series_str
                
                sample = {
                    'id': row['id'],
                    'frequency': frequency,
                    'series': series_list,
                    'caption': row['caption']
                }
                all_samples.append(sample)
            except Exception as e:
                print(f"Warning: Could not parse series for {row['id']}: {e}")
                continue
    
    # Create dataset
    full_dataset = Dataset.from_list(all_samples)
    
    # Split into train/val/test
    train_val, test = full_dataset.train_test_split(
        test_size=TEST_FRAC, seed=seed
    ).values()
    
    val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
    train, val = train_val.train_test_split(
        test_size=val_frac_adj, seed=seed + 1
    ).values()
    
    print(f"Dataset splits - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    return train, val, test

# ---------------------------
# Helper functions
# ---------------------------

def get_frequency_distribution(dataset: Dataset) -> Dict[str, int]:
    """
    Get the distribution of frequencies in a dataset.
    
    Args:
        dataset: The dataset to analyze.
    Returns:
        Dictionary mapping frequency to count.
    """
    frequencies = dataset['frequency']
    return dict(pd.Series(frequencies).value_counts())

def print_dataset_info(dataset: Dataset, name: str):
    """
    Print information about a dataset split.
    
    Args:
        dataset: The dataset split.
        name: Name of the split (e.g., 'Train').
    """
    freq_dist = get_frequency_distribution(dataset)
    print(f"\n{name} dataset:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Frequency distribution:")
    for freq, count in freq_dist.items():
        print(f"    {freq}: {count} ({count/len(dataset)*100:.1f}%)")

# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Load all data
    data_dict = load_all_m4_data()
    
    # Create splits
    train, val, test = create_combined_dataset(data_dict)
    
    # Print information
    print_dataset_info(train, "Train")
    print_dataset_info(val, "Validation")
    print_dataset_info(test, "Test")
    
    # Example of accessing data
    print(f"\nExample sample from train:")
    sample = train[0]
    print(f"  ID: {sample['id']}")
    print(f"  Frequency: {sample['frequency']}")
    print(f"  Series length: {len(sample['series'])}")
    print(f"  Caption preview: {sample['caption'][:100]}...")