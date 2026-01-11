# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import os
import json
import pandas as pd
from typing import Tuple, Dict, List, Optional
from datasets import Dataset
from tqdm import tqdm

from opentslm.time_series_datasets.constants import RAW_DATA as RAW_DATA_PATH


# MimicIV-ECG Directory - completely independent from ECG-QA
# The data is located at /home/lastchance/Desktop/mimiciv
MIMICIV_ECG_BASE_DIR = "/home/lastchance/Desktop/mimiciv/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"

# Default path for ECG data files (.dat/.hea) - same directory or subdirectory
DEFAULT_MIMICIV_ECG_DATA_DIR = MIMICIV_ECG_BASE_DIR


def get_mimiciv_ecg_path(ecg_id: int, ecg_data_dir: Optional[str] = None) -> str:
    """
    Get the file path for a MimicIV-ECG record.
    
    MimicIV-ECG structure: files/p{patient}/p{patient}/s{session}/{ecg_id}.dat
    
    Args:
        ecg_id: The ECG record ID
        ecg_data_dir: Optional directory containing ECG data files. 
                     If None, uses DEFAULT_MIMICIV_ECG_DATA_DIR
    
    Returns:
        Base path to the ECG record (without .dat/.hea extension)
    """
    if ecg_data_dir is None:
        ecg_data_dir = DEFAULT_MIMICIV_ECG_DATA_DIR
    
    files_dir = os.path.join(ecg_data_dir, "files")
    
    # Search for the ECG file by walking the directory structure
    # Format: files/p{patient}/p{patient}/s{session}/{ecg_id}.dat
    ecg_id_str = str(ecg_id)
    for root, dirs, files in os.walk(files_dir):
        for file in files:
            if file == f"{ecg_id_str}.dat":
                base_path = os.path.join(root, ecg_id_str)
                if os.path.exists(base_path + ".hea"):
                    return base_path
    
    # If not found, return a path that will fail with a clear error
    return os.path.join(files_dir, "not_found", str(ecg_id))


def load_mimiciv_ecg_splits(
    ecg_data_dir: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load all MimicIV-ECG files by scanning for ECG files.
    
    Since we use custom prompts, we don't need JSON files with questions.
    We just scan for all available ECG files and create samples from them.
    Returns all samples as a single dataset (same for train/val/test).
    
    Args:
        ecg_data_dir: Optional directory containing ECG data files.
                     If None, uses DEFAULT_MIMICIV_ECG_DATA_DIR
        max_samples: Optional maximum number of samples (for testing)
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) - all contain the same data
    """
    if ecg_data_dir is None:
        ecg_data_dir = DEFAULT_MIMICIV_ECG_DATA_DIR
    
    if not os.path.exists(ecg_data_dir):
        raise FileNotFoundError(
            f"MimicIV-ECG data directory not found at {ecg_data_dir}. "
            f"Please ensure the data is available."
        )
    
    # Find all ECG .dat files
    files_dir = os.path.join(ecg_data_dir, "files")
    if not os.path.exists(files_dir):
        raise FileNotFoundError(
            f"ECG files directory not found at {files_dir}. "
            f"Expected structure: {ecg_data_dir}/files/p*/p*/s*/*.dat"
        )
    
    print("Scanning for ECG files...")
    ecg_files = []
    for root, dirs, files in os.walk(files_dir):
        for file in files:
            if file.endswith('.dat'):
                base_path = os.path.join(root, file.replace('.dat', ''))
                hea_path = base_path + '.hea'
                if os.path.exists(hea_path):
                    ecg_files.append(base_path)
    
    print(f"Found {len(ecg_files)} ECG files")
    
    if not ecg_files:
        raise ValueError(f"No ECG files found in {files_dir}")
    
    # Extract ECG ID from file path (e.g., .../s46394165/46394165 -> 46394165)
    def extract_ecg_id(file_path: str) -> int:
        """Extract ECG ID from file path."""
        # Path format: .../s{ecg_id}/{ecg_id}
        parts = file_path.split('/')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        # Fallback: use filename
        filename = os.path.basename(file_path)
        if filename.isdigit():
            return int(filename)
        raise ValueError(f"Could not extract ECG ID from path: {file_path}")
    
    # Create samples from ECG files
    all_samples = []
    for ecg_file in tqdm(ecg_files, desc="Creating samples"):
        try:
            ecg_id = extract_ecg_id(ecg_file)
            sample = {
                'ecg_id': [ecg_id],
                'ecg_paths': [ecg_file + '.dat'],
                'sample_id': len(all_samples),
            }
            all_samples.append(sample)
        except Exception as e:
            print(f"Warning: Skipping {ecg_file}: {e}")
            continue
    
    # Limit samples if requested
    if max_samples and len(all_samples) > max_samples:
        print(f"Limiting to {max_samples} samples...")
        all_samples = all_samples[:max_samples]
    
    print(f"Total samples: {len(all_samples)}")
    
    # Convert to HuggingFace dataset - return same dataset for all splits
    print("Converting to HuggingFace dataset...")
    dataset = Dataset.from_list(all_samples)
    
    print("Dataset loading complete!")
    return dataset, dataset, dataset


def load_mimiciv_ecg_answers() -> pd.DataFrame:
    """Load the answers mapping for MimicIV-ECG."""
    if not os.path.exists(MIMICIV_ECG_BASE_DIR):
        raise FileNotFoundError(
            f"MimicIV-ECG directory not found at {MIMICIV_ECG_BASE_DIR}"
        )
    
    answers_path = os.path.join(MIMICIV_ECG_BASE_DIR, "answers.csv")
    if not os.path.exists(answers_path):
        raise FileNotFoundError(f"Answers file not found: {answers_path}")
    
    return pd.read_csv(answers_path)


if __name__ == "__main__":
    # Test the loader
    print("Testing MimicIV-ECG loader...")
    
    try:
        train, val, test = load_mimiciv_ecg_splits(max_samples=10)
        print(f"Loaded MimicIV-ECG dataset:")
        print(f"  Train: {len(train)} samples")
        print(f"  Validation: {len(val)} samples")
        print(f"  Test: {len(test)} samples")
        
        if len(train) > 0:
            print(f"\nSample from training set:")
            sample = train[0]
            for key, value in sample.items():
                if isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: {value[:3]}... ({len(value)} items)")
                else:
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
