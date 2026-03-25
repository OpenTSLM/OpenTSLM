# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT


import os
import subprocess
import json
import requests
import shutil
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from datasets import Dataset
import os

from opentslm.time_series_datasets.constants import RAW_DATA as RAW_DATA_PATH
from tqdm import tqdm


# ECG-QA Repository
ECG_QA_URL = "https://github.com/Jwoo5/ecg-qa"
ECG_QA_DIR = os.path.join(RAW_DATA_PATH, "ecg_qa")

# PTB-XL Dataset  
PTBXL_ZIP_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"
PTBXL_DIR = os.path.join(RAW_DATA_PATH, "ptbxl")
PTBXL_RECORDS_DIR = os.path.join(PTBXL_DIR, "records500")


def ensure_directory_exists(directory: str):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def does_ecg_qa_exist():
    """Check if ECG-QA repository exists locally."""
    return os.path.exists(ECG_QA_DIR) and os.path.exists(os.path.join(ECG_QA_DIR, "ecgqa"))


def does_ptbxl_exist():
    """Check if PTB-XL dataset exists locally."""
    return (os.path.exists(PTBXL_DIR) and 
            os.path.exists(PTBXL_RECORDS_DIR) and
            os.path.exists(os.path.join(PTBXL_DIR, "ptbxl_database.csv")))


def clone_ecg_qa():
    """Clone the ECG-QA repository."""
    ensure_directory_exists(RAW_DATA_PATH)
    print(f"Cloning ECG-QA repository into {ECG_QA_DIR}...")
    subprocess.run([
        "git", "clone", ECG_QA_URL, ECG_QA_DIR
    ], check=True)
    print("ECG-QA repository cloned successfully!")


def download_ptbxl():
    """Download PTB-XL dataset from PhysioNet."""
    ensure_directory_exists(RAW_DATA_PATH)
    ptbxl_zip_path = os.path.join(RAW_DATA_PATH, "ptb-xl-1.0.3.zip")
    
    print(f"Downloading PTB-XL dataset to {PTBXL_DIR}...")
    
    # Download the zip file if it doesn't exist
    if not os.path.exists(ptbxl_zip_path):
        print(f"Downloading PTB-XL zip file from {PTBXL_ZIP_URL}...")
        
        try:
            # Use wget for large files (more reliable for big downloads)
            subprocess.run([
                "wget", "-O", ptbxl_zip_path, PTBXL_ZIP_URL
            ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to Python requests if wget is not available
            print("wget not available, using Python requests (this may be slow for large files)...")
            response = requests.get(PTBXL_ZIP_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(ptbxl_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end="", flush=True)
            print()  # New line after progress
    
    # Extract the zip file if the target directory doesn't exist
    if not os.path.exists(PTBXL_DIR):
        print("Extracting PTB-XL dataset...")
        import zipfile
        
        with zipfile.ZipFile(ptbxl_zip_path, 'r') as zip_ref:
            # Extract to a temporary directory first
            temp_extract_dir = os.path.join(RAW_DATA_PATH, "temp_ptbxl_extract")
            
            # Get list of files to extract
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            print(f"Extracting {total_files} files from PTB-XL dataset...")
            from tqdm import tqdm
            
            # Extract with progress bar
            for file_info in tqdm(file_list, desc="Extracting PTB-XL", unit="files"):
                zip_ref.extract(file_info, temp_extract_dir)
            
            # Find the actual ptb-xl directory (it might be nested)
            extracted_dirs = [d for d in os.listdir(temp_extract_dir) 
                            if os.path.isdir(os.path.join(temp_extract_dir, d))]
            
            if extracted_dirs:
                # Move the first directory (should be ptb-xl-1.0.3 or similar) to our target
                source_dir = os.path.join(temp_extract_dir, extracted_dirs[0])
                shutil.move(source_dir, PTBXL_DIR)
                
                # Clean up temp directory
                shutil.rmtree(temp_extract_dir)
            else:
                # If no subdirectory, move the temp directory itself
                shutil.move(temp_extract_dir, PTBXL_DIR)
        
        print("PTB-XL dataset extracted successfully!")
    
    # Verify that key files exist
    required_files = ["ptbxl_database.csv", "scp_statements.csv"]
    for req_file in required_files:
        file_path = os.path.join(PTBXL_DIR, req_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required PTB-XL file {req_file} not found in {PTBXL_DIR}. "
                                   f"PTB-XL dataset may be corrupted or incomplete.")
    
    if not os.path.exists(PTBXL_RECORDS_DIR):
        # List what's actually in the directory to help debug
        contents = []
        if os.path.exists(PTBXL_DIR):
            contents = [f"  {item}" for item in os.listdir(PTBXL_DIR)]
            contents_str = "\n".join(contents)
        else:
            contents_str = "  Directory does not exist"
        
        raise FileNotFoundError(f"PTB-XL records directory not found at {PTBXL_RECORDS_DIR}. "
                               f"PTB-XL dataset may be corrupted or incomplete.\n"
                               f"Contents of {PTBXL_DIR}:\n{contents_str}")


def download_ecg_qa_if_not_exists():
    """Download ECG-QA repository if it doesn't exist."""
    if not does_ecg_qa_exist():
        clone_ecg_qa()


def download_ptbxl_if_not_exists():
    """Download PTB-XL dataset if it doesn't exist."""
    if not does_ptbxl_exist():
        download_ptbxl()


def get_ptbxl_ecg_path(ecg_id: int) -> str:
    """Get the file path for a PTB-XL ECG record."""
    return os.path.join(
        PTBXL_RECORDS_DIR,
        f"{int(ecg_id / 1000) * 1000:05d}",
        f"{ecg_id:05d}_hr"
    )


def load_ptbxl_metadata() -> pd.DataFrame:
    """Load PTB-XL metadata for safe clinical context."""
    download_ptbxl_if_not_exists()
    
    ptbxl_path = os.path.join(PTBXL_DIR, "ptbxl_database.csv")
    if not os.path.exists(ptbxl_path):
        raise FileNotFoundError(f"PTB-XL database not found: {ptbxl_path}")
    
    print("Loading PTB-XL metadata...")
    df = pd.read_csv(ptbxl_path)
    
    # Select only safe columns (no diagnostic information)
    safe_columns = [
        'ecg_id', 'patient_id', 'age', 'sex', 'height', 'weight',
        'recording_date', 'device', 'site', 'nurse',
        'baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems',
        'extra_beats', 'pacemaker', 'validated_by_human'
    ]
    
    # Keep only columns that exist in the dataframe
    available_columns = [col for col in safe_columns if col in df.columns]
    metadata_df = df[available_columns].copy()
    
    print(f"Loaded metadata for {len(metadata_df)} ECG records")
    return metadata_df


def create_clinical_context(ecg_id: int, ptbxl_metadata: pd.DataFrame) -> str:
    """Create safe clinical context from PTB-XL metadata without diagnostic spoilers."""
    
    # Find the ECG record
    record = ptbxl_metadata[ptbxl_metadata['ecg_id'] == ecg_id]
    
    if record.empty:
        return "12-lead ECG recording. Signal quality adequate for analysis."
    
    record = record.iloc[0]  # Get first (should be only) match
    
    # Build context from safe metadata
    context_parts = []
    
    # Patient demographics (safe medical context)
    if pd.notna(record.get('age')):
        age = int(record['age'])
        context_parts.append(f"{age}-year-old")
    
    if pd.notna(record.get('sex')):
        sex = "male" if record['sex'] == 1 else "female"
        context_parts.append(f"{sex} patient")
    
    # Recording technical details
    context_parts.append("12-lead ECG")
    
    if pd.notna(record.get('recording_date')):
        # Just mention it was recorded, don't give exact date for privacy
        context_parts.append("clinical recording")
    
    if pd.notna(record.get('device')):
        device = str(record['device']).strip()
        if device and device != 'nan':
            context_parts.append(f"recorded with {device}")
    
    # Signal quality information (helpful technical context)
    quality_issues = []
    if record.get('baseline_drift') and str(record['baseline_drift']).strip():
        quality_issues.append("baseline drift noted")
    if record.get('static_noise') and str(record['static_noise']).strip():
        quality_issues.append("static noise present")
    if record.get('burst_noise') and str(record['burst_noise']).strip():
        quality_issues.append("burst noise present")
    if record.get('electrodes_problems') and str(record['electrodes_problems']).strip():
        quality_issues.append("electrode artifacts present")
    
    if quality_issues:
        context_parts.append(f"Signal quality: {', '.join(quality_issues)}")
    else:
        context_parts.append("Signal quality: adequate for analysis")
    
    # Additional technical context
    if record.get('extra_beats'):
        context_parts.append("extra beats detected during recording")
    
    if record.get('pacemaker'):
        context_parts.append("pacemaker present")
    
    # Combine into natural sentence
    if len(context_parts) >= 2:
        context = f"{context_parts[0]} {context_parts[1]}. " + ". ".join(context_parts[2:]) + "."
    else:
        context = ". ".join(context_parts) + "."
    
    return context


def load_ecg_qa_ptbxl_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """Load ECG-QA PTB-XL splits as HuggingFace datasets."""
    
    # Ensure both datasets exist
    download_ecg_qa_if_not_exists()
    download_ptbxl_if_not_exists()
    
    ptbxl_ecgqa_dir = os.path.join(ECG_QA_DIR, "ecgqa", "ptbxl")
    
    if not os.path.exists(ptbxl_ecgqa_dir):
        raise FileNotFoundError(f"PTB-XL ECG-QA directory not found at {ptbxl_ecgqa_dir}")
    
    # Load the template version (paraphrased is also available)
    template_dir = os.path.join(ptbxl_ecgqa_dir, "template")
    
    def load_split_data(split_name: str) -> List[Dict]:
        """Load data for a specific split."""
        print(f"Loading {split_name} split...")
        split_dir = os.path.join(template_dir, split_name)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        all_data = []
        
        # Get all JSON files
        json_files = [f for f in os.listdir(split_dir) if f.endswith('.json')]
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in split directory: {split_dir}")
        
        # Load PTB-XL metadata once for this split
        ptbxl_metadata = load_ptbxl_metadata()
        
        # Load all JSON files in the split directory with progress bar
        for json_file in tqdm(json_files, desc=f"Loading {split_name} JSON files"):
            json_path = os.path.join(split_dir, json_file)
            
            try:
                with open(json_path, 'r') as f:
                    split_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file {json_path}: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Failed to read JSON file {json_path}: {str(e)}")
            
            if not isinstance(split_data, list):
                raise ValueError(f"Expected JSON file {json_path} to contain a list, got {type(split_data)}")
            
            # Add ECG file paths and clinical context to each sample
            for sample_idx, sample in enumerate(tqdm(split_data, desc=f"Processing {json_file}", leave=False)):
                if not isinstance(sample, dict):
                    raise ValueError(f"Expected sample {sample_idx} in {json_path} to be a dict, got {type(sample)}")
                
                if "ecg_id" not in sample:
                    raise KeyError(f"Sample {sample_idx} in {json_path} missing required 'ecg_id' field")
                
                sample['ecg_paths'] = []
                sample['clinical_contexts'] = []
                ecg_ids = sample['ecg_id']
                
                if not isinstance(ecg_ids, list):
                    raise ValueError(f"Expected 'ecg_id' in sample {sample_idx} of {json_path} to be a list, got {type(ecg_ids)}")
                
                if not ecg_ids:
                    raise ValueError(f"Sample {sample_idx} in {json_path} has empty ecg_id list. "
                                   f"Every ECG-QA sample must have at least one ECG ID. "
                                   f"This indicates corrupted or invalid data.")
                
                for ecg_id in ecg_ids:
                    if not isinstance(ecg_id, int):
                        raise ValueError(f"Expected ECG ID to be integer, got {type(ecg_id)}: {ecg_id}")
                    
                    ecg_path = get_ptbxl_ecg_path(ecg_id)
                    # Check if the ECG file exists
                    if os.path.exists(ecg_path + '.dat'):
                        sample['ecg_paths'].append(ecg_path + '.dat')
                    else:
                        # Missing ECG files are a critical error - the dataset is incomplete
                        raise FileNotFoundError(f"ECG file not found: {ecg_path}.dat (ECG ID: {ecg_id}). "
                                              f"This indicates that the PTB-XL dataset is incomplete or corrupted. "
                                              f"Please re-download the PTB-XL dataset or check the ECG ID {ecg_id} "
                                              f"in sample {sample_idx} of {json_path}.")
                    
                    # Add safe clinical context
                    clinical_context = create_clinical_context(ecg_id, ptbxl_metadata)
                    sample['clinical_contexts'].append(clinical_context)
            
            all_data.extend(split_data)
        
        print(f"Loaded {len(all_data)} samples for {split_name} split")
        return all_data
    
    # Load each split
    train_data = load_split_data("train")
    val_data = load_split_data("valid") 
    test_data = load_split_data("test")
    
    # Convert to HuggingFace datasets with progress
    print("Converting to HuggingFace datasets...")
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    print("Dataset loading complete!")
    return train_dataset, val_dataset, test_dataset


def load_ecg_qa_answers() -> pd.DataFrame:
    """Load the answers mapping for ECG-QA."""
    download_ecg_qa_if_not_exists()
    
    answers_path = os.path.join(ECG_QA_DIR, "ecgqa", "ptbxl", "answers.csv")
    if not os.path.exists(answers_path):
        raise FileNotFoundError(f"Answers file not found: {answers_path}")
    
    return pd.read_csv(answers_path)


if __name__ == "__main__":
    # Test the loader
    print("Testing ECG-QA loader with clinical context...")
    
    # Test individual components
    print(f"ECG-QA exists: {does_ecg_qa_exist()}")
    print(f"PTB-XL exists: {does_ptbxl_exist()}")
    
    try:
        train, val, test = load_ecg_qa_ptbxl_splits()
        print(f"Loaded ECG-QA PTB-XL dataset:")
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
                    
            # Show clinical context
            if 'clinical_contexts' in sample and sample['clinical_contexts']:
                print(f"\nClinical context: {sample['clinical_contexts'][0]}")
                    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

