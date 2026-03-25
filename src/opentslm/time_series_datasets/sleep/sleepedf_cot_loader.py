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
from opentslm.time_series_datasets.constants import RAW_DATA
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

SLEEP_DATA_DIR = os.path.join(RAW_DATA, "sleep")
COT_CSV = os.path.join(SLEEP_DATA_DIR, "sleep_cot.csv")
SLEEPEDF_RELEASE_URL = "https://polybox.ethz.ch/index.php/s/ZryWSdCFJZ9JR3R/download"

TEST_FRAC = 0.1
VAL_FRAC = 0.1

def download_and_extract_sleepedf():
    """
    Download the SleepEDF CoT CSV into data/sleep/ if not already present, with a progress bar.
    """
    if os.path.exists(SLEEP_DATA_DIR) and os.path.exists(COT_CSV):
        print(f"SleepEDF CoT dataset already exists at {COT_CSV}")
        return
    os.makedirs(SLEEP_DATA_DIR, exist_ok=True)
    print(f"Downloading SleepEDF CoT dataset from {SLEEPEDF_RELEASE_URL}...")
    try:
        with urllib.request.urlopen(SLEEPEDF_RELEASE_URL) as response:
            total = int(response.headers.get('content-length', 0))
            with open(COT_CSV, "wb") as f, tqdm(
                total=total, unit='B', unit_scale=True, desc="Downloading sleep_cot.csv"
            ) as pbar:
                for chunk in iter(lambda: response.read(8192), b""):
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("Download completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to download SleepEDF CoT dataset: {e}")
    if not os.path.exists(COT_CSV):
        raise FileNotFoundError(f"sleep_cot.csv not found after download in {SLEEP_DATA_DIR}")

def ensure_sleepedf_cot_dataset():
    """
    Ensure the SleepEDF CoT dataset is available in data/sleep/.
    Download if necessary.
    """
    if not os.path.exists(COT_CSV):
        download_and_extract_sleepedf()

def load_sleepedf_cot_splits(seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the SleepEDF CoT dataset and split it into train, validation, and test sets.
    Uses stratified splitting to ensure all classes are represented in each split.
    Args:
        seed: Random seed for reproducible splits
    Returns:
        Tuple of (train, validation, test) datasets
    """
    ensure_sleepedf_cot_dataset()
    if not os.path.exists(COT_CSV):
        raise FileNotFoundError(f"CoT CSV not found: {COT_CSV}")
    df = pd.read_csv(COT_CSV)
    def parse_series(s):
        try:
            parsed = ast.literal_eval(s)
            # Extract the first element (the actual time series)
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list):
                assert len(parsed[0]) == 1500, f"Expected 1500 elements, got {len(parsed[0])}"
                return parsed[0]
            else:
                raise ValueError(f"Expected format [[...]], got: {parsed[:10] if isinstance(parsed, list) else type(parsed)}")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse time series: {e}")
    if 'time_series' in df.columns:
        print("Parsing SleepEDF CoT data (this may take a while)...")
        df['time_series'] = [parse_series(s) for s in tqdm(df['time_series'], desc='Parsing SleepEDF CoT data')]
    # Stratified split by label
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_FRAC,
        random_state=seed,
        stratify=df['label']
    )
    val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_frac_adj,
        random_state=seed+1,
        stratify=train_val_df['label']
    )
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    return train_dataset, val_dataset, test_dataset

def get_label_distribution(dataset: Dataset) -> Dict[str, int]:
    labels = dataset['label']
    return dict(pd.Series(labels).value_counts())

def print_dataset_info(dataset: Dataset, name: str):
    label_dist = get_label_distribution(dataset)
    print(f"\n{name} dataset:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Label distribution:")
    for label, count in label_dist.items():
        print(f"    {label}: {count} ({count/len(dataset)*100:.1f}%)")

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_sleepedf_cot_splits()
    print_dataset_info(train_ds, "Train")
    print_dataset_info(val_ds, "Validation")
    print_dataset_info(test_ds, "Test")
    if len(train_ds) > 0:
        sample = train_ds[0]
        print("\nSample data:")
        for key, value in sample.items():
            if key == 'time_series':
                if isinstance(value, list) and len(value) > 0:
                    print(f"{key}: {value[:5]}... (truncated)")
                else:
                    print(f"{key}: {value}")
            else:
                print(f"{key}: {value}") 