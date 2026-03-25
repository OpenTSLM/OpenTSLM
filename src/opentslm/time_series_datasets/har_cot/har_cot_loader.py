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
from opentslm.time_series_datasets.constants import RAW_DATA
from tqdm.auto import tqdm
import logging
from opentslm.logger import get_logger

HAR_COT_DATA_DIR = os.path.join(RAW_DATA, "har_cot")
HAR_COT_ZIP = os.path.join(HAR_COT_DATA_DIR, "har_cot.zip")
HAR_COT_TRAIN_CSV = os.path.join(HAR_COT_DATA_DIR, "har_cot_train_cot.csv")
HAR_COT_VAL_CSV = os.path.join(HAR_COT_DATA_DIR, "har_cot_val_cot.csv")
HAR_COT_TEST_CSV = os.path.join(HAR_COT_DATA_DIR, "har_cot_test_cot.csv")
HAR_COT_RELEASE_URL = "https://polybox.ethz.ch/index.php/s/kD74GnMYxn3HBEM/download"

def download_and_extract_har_cot():
    """
    Download the HAR CoT dataset zip file and extract the CSV files if not already present.
    """
    # Check if all CSV files already exist
    if (os.path.exists(HAR_COT_TRAIN_CSV) and 
        os.path.exists(HAR_COT_VAL_CSV) and 
        os.path.exists(HAR_COT_TEST_CSV)):
        print(f"HAR CoT dataset already exists at {HAR_COT_DATA_DIR}")
        return

    os.makedirs(HAR_COT_DATA_DIR, exist_ok=True)
    
    # Download the zip file if it doesn't exist
    if not os.path.exists(HAR_COT_ZIP):
        print(f"Downloading HAR CoT dataset from {HAR_COT_RELEASE_URL}...")
        try:
            with urllib.request.urlopen(HAR_COT_RELEASE_URL) as response:
                total = int(response.headers.get('content-length', 0))
                with open(HAR_COT_ZIP, "wb") as f, tqdm(
                    total=total, unit='B', unit_scale=True, desc="Downloading har_cot.zip"
                ) as pbar:
                    for chunk in iter(lambda: response.read(8192), b""):
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            print("Download completed successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download HAR CoT dataset: {e}")
    
    # Extract the zip file
    print("Extracting HAR CoT dataset...")
    try:
        with zipfile.ZipFile(HAR_COT_ZIP, 'r') as zip_ref:
            zip_ref.extractall(HAR_COT_DATA_DIR)
        print("Extraction completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract HAR CoT dataset: {e}")
    
    # Verify all CSV files exist
    if not all([os.path.exists(HAR_COT_TRAIN_CSV), 
                os.path.exists(HAR_COT_VAL_CSV), 
                os.path.exists(HAR_COT_TEST_CSV)]):
        raise FileNotFoundError(f"CSV files not found after extraction in {HAR_COT_DATA_DIR}")

def ensure_har_cot_dataset():
    """
    Ensure the HAR CoT dataset is available in data/har_cot/.
    Download and extract if necessary.
    """
    if not (os.path.exists(HAR_COT_TRAIN_CSV) and 
            os.path.exists(HAR_COT_VAL_CSV) and 
            os.path.exists(HAR_COT_TEST_CSV)):
        download_and_extract_har_cot()

def parse_time_series(series_str):
    """
    Parse the time series string from the CSV into a list of floats.
    
    Args:
        series_str: String representation of the time series data
        
    Returns:
        List of float values
    """
    try:
        # Use ast.literal_eval to safely parse the list string
        return ast.literal_eval(series_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing time series: {e}")
        print(f"String: {series_str[:100]}...")
        raise

def load_har_cot_csv(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess a HAR CoT CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Processed DataFrame with parsed time series data
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter out rows that are actually headers (where x_axis == 'x_axis')
    initial_size = len(df)
    df = df[df['x_axis'] != 'x_axis']
    df = df.reset_index(drop=True)
    removed_rows = initial_size - len(df)
    
    if removed_rows > 0:
        print(f"Removed {removed_rows} header rows from the data")
    
    # Parse the time series columns with progress bars
    print("Parsing time series data...")
    
    tqdm.pandas(desc="Parsing x_axis")
    df['x_axis'] = df['x_axis'].progress_apply(parse_time_series)
    
    tqdm.pandas(desc="Parsing y_axis")
    df['y_axis'] = df['y_axis'].progress_apply(parse_time_series)
    
    tqdm.pandas(desc="Parsing z_axis")
    df['z_axis'] = df['z_axis'].progress_apply(parse_time_series)
    
    print(f"Loaded {len(df)} samples with {df['label'].nunique()} unique labels")
    return df

def load_har_cot_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the HAR CoT dataset splits.
    
    Returns:
        Tuple of (train, validation, test) Dataset objects
    """
    # Ensure dataset is available
    ensure_har_cot_dataset()
    
    # Load the three splits
    train_df = load_har_cot_csv(HAR_COT_TRAIN_CSV)
    val_df = load_har_cot_csv(HAR_COT_VAL_CSV)
    test_df = load_har_cot_csv(HAR_COT_TEST_CSV)
    
    # Convert to Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, val_dataset, test_dataset

def get_label_distribution(dataset: Dataset) -> Dict[str, int]:
    """
    Get label distribution for a dataset.
    
    Args:
        dataset: Dataset object
        
    Returns:
        Dictionary mapping labels to counts
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
    for label, count in sorted(label_dist.items()):
        print(f"    {label}: {count} ({count/len(dataset)*100:.1f}%)")

if __name__ == "__main__":
    print("=== HAR CoT Dataset Loading Demo ===\n")
    
    # Load the dataset splits
    train_ds, val_ds, test_ds = load_har_cot_splits()
    
    # Print dataset information
    print_dataset_info(train_ds, "Train")
    print_dataset_info(val_ds, "Validation") 
    print_dataset_info(test_ds, "Test")
    
    # Show sample data
    if len(train_ds) > 0:
        print("\n" + "="*50 + "\n")
        print("Sample data from training set:")
        sample = train_ds[0]
        for key, value in sample.items():
            if key in ['x_axis', 'y_axis', 'z_axis']:
                if isinstance(value, list) and len(value) > 0:
                    print(f"{key}: {value[:5]}... (length: {len(value)})")
                else:
                    print(f"{key}: {value}")
            elif key == 'rationale':
                print(f"{key}: {value[:100]}..." if len(str(value)) > 100 else f"{key}: {value}")
            else:
                print(f"{key}: {value}") 