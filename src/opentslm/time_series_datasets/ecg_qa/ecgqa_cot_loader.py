# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import subprocess
import requests
import shutil
from typing import Tuple, Dict, List
from datasets import Dataset
import os
import zipfile

from opentslm.time_series_datasets.constants import RAW_DATA as RAW_DATA_PATH
from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import (
    download_ecg_qa_if_not_exists,
    download_ptbxl_if_not_exists
)
from tqdm import tqdm

# ECG-QA CoT Repository
ECG_QA_COT_URL = "https://polybox.ethz.ch/index.php/s/D5QaJSEw4dXkzXm/download/ecg_qa_cot_final.zip"
ECG_QA_COT_DIR = os.path.join(RAW_DATA_PATH, "ecg_qa_cot")
ECG_QA_COT_ZIP = "ecg_qa_cot.zip"


def ensure_directory_exists(directory: str):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def does_ecg_qa_cot_exist():
    """Check if ECG-QA CoT data exists locally."""
    return os.path.exists(ECG_QA_COT_DIR) and os.path.exists(os.path.join(ECG_QA_COT_DIR, "ecg_qa_cot_train.csv"))


def download_ecg_qa_cot():
    """Download ECG-QA CoT data from polybox."""
    ensure_directory_exists(RAW_DATA_PATH)
    cot_zip_path = os.path.join(RAW_DATA_PATH, ECG_QA_COT_ZIP)
    
    print(f"Downloading ECG-QA CoT data to {ECG_QA_COT_DIR}...")
    
    # Download the zip file if it doesn't exist
    if not os.path.exists(cot_zip_path):
        print(f"Downloading ECG-QA CoT zip file from {ECG_QA_COT_URL}...")
        
        try:
            # Use wget for large files (more reliable for big downloads)
            subprocess.run([
                "wget", "-O", cot_zip_path, ECG_QA_COT_URL
            ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to Python requests if wget is not available
            print("wget not available, using Python requests (this may be slow for large files)...")
            response = requests.get(ECG_QA_COT_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(cot_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end="", flush=True)
            print()  # New line after progress
    
    # Extract the zip file if the target directory doesn't exist
    if not os.path.exists(ECG_QA_COT_DIR):
        print("Extracting ECG-QA CoT data...")
        
        with zipfile.ZipFile(cot_zip_path, 'r') as zip_ref:
            # Extract to a temporary directory first
            temp_extract_dir = os.path.join(RAW_DATA_PATH, "temp_ecgqa_cot_extract")
            
            # Get list of files to extract
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            print(f"Extracting {total_files} files from ECG-QA CoT dataset...")
            from tqdm import tqdm
            
            # Extract with progress bar
            for file_info in tqdm(file_list, desc="Extracting ECG-QA CoT", unit="files"):
                zip_ref.extract(file_info, temp_extract_dir)
            
            # Find the actual data directory (it might be nested)
            extracted_dirs = [d for d in os.listdir(temp_extract_dir) 
                            if os.path.isdir(os.path.join(temp_extract_dir, d))]
            
            if extracted_dirs:
                # Move the first directory to our target
                source_dir = os.path.join(temp_extract_dir, extracted_dirs[0])
                shutil.move(source_dir, ECG_QA_COT_DIR)
                
                # Clean up temp directory
                shutil.rmtree(temp_extract_dir)
            else:
                # If no subdirectory, move the temp directory itself
                shutil.move(temp_extract_dir, ECG_QA_COT_DIR)
        
        print("ECG-QA CoT data extracted successfully!")
    
    # Verify that key files exist
    train_file = os.path.join(ECG_QA_COT_DIR, "ecg_qa_cot_train.csv")
    val_file = os.path.join(ECG_QA_COT_DIR, "ecg_qa_cot_val.csv")
    test_file = os.path.join(ECG_QA_COT_DIR, "ecg_qa_cot_test.csv")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"ECG-QA CoT train file not found: {train_file}")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"ECG-QA CoT validation file not found: {val_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"ECG-QA CoT test file not found: {test_file}")


def download_ecg_qa_cot_if_not_exists():
    """Download ECG-QA CoT data if it doesn't exist."""
    if not does_ecg_qa_cot_exist():
        download_ecg_qa_cot()


def load_ecg_qa_cot_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """Load ECG-QA CoT dataset splits directly from CoT CSVs and PTB-XL files.

    This loader is independent from the ECG-QA template loader. It reads the
    CoT CSVs (train/val/test), parses the required fields, resolves PTB-XL
    file paths for each ecg_id, and constructs HF datasets with:
      - question, answer, template_id, question_type
      - ecg_id (list[int]), ecg_paths (list[str]), clinical_contexts (list[str])
      - rationale (string)
    """

    # Ensure datasets exist
    download_ecg_qa_if_not_exists()
    download_ptbxl_if_not_exists()
    download_ecg_qa_cot_if_not_exists()

    import pandas as pd
    from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import get_ptbxl_ecg_path

    def parse_ecg_id(ecg_id_raw: str) -> int:
        if ecg_id_raw is None:
            raise ValueError("Missing ecg_id in CoT CSV row")
        try:
            ecg_id_clean = str(ecg_id_raw).strip().strip("[]").strip()
            return int(ecg_id_clean)
        except Exception as e:
            raise ValueError(f"Failed to parse ecg_id '{ecg_id_raw}': {e}")

    def load_split_from_csv(split_name: str) -> List[Dict]:
        if split_name == "train":
            split_file = os.path.join(ECG_QA_COT_DIR, "ecg_qa_cot_train.csv")
        elif split_name == "validation":
            split_file = os.path.join(ECG_QA_COT_DIR, "ecg_qa_cot_val.csv")
        elif split_name == "test":
            split_file = os.path.join(ECG_QA_COT_DIR, "ecg_qa_cot_test.csv")
        else:
            raise ValueError(f"Unknown split name: {split_name}")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"CoT split file not found: {split_file}")

        print(f"Loading CoT data for {split_name} split from {split_file}...")
        print("Reading CSV file...")
        df = pd.read_csv(split_file)
        print(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")

        out: List[Dict] = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name} CoT rows"):
            question = row.get("question")
            answer = row.get("answer")
            template_id = row.get("template_id")
            question_type = row.get("question_type")
            clinical_context = row.get("clinical_context")
            rationale = row.get("rationale")
            ecg_id_val = parse_ecg_id(row.get("ecg_id"))

            # Validate required fields
            if question is None or not isinstance(question, str) or question.strip() == "":
                raise ValueError(f"Missing/invalid question at row {idx}")
            if answer is None or not isinstance(answer, str) or answer.strip() == "":
                raise ValueError(f"Missing/invalid answer at row {idx}")
            if template_id is None:
                raise ValueError(f"Missing template_id at row {idx}")
            if question_type is None or not isinstance(question_type, str) or question_type.strip() == "":
                raise ValueError(f"Missing/invalid question_type at row {idx}")
            if clinical_context is None:
                clinical_context = "12-lead ECG recording."
            if rationale is None or not isinstance(rationale, str) or rationale.strip() == "":
                raise ValueError(f"Missing/invalid rationale at row {idx}")

            # Resolve PTB-XL file path
            ecg_base = get_ptbxl_ecg_path(ecg_id_val)
            dat_path = ecg_base + ".dat"
            hea_path = ecg_base + ".hea"
            if not os.path.exists(dat_path) or not os.path.exists(hea_path):
                raise FileNotFoundError(
                    f"PTB-XL files not found for ecg_id {ecg_id_val}: {dat_path}, {hea_path}")

            sample = {
                "question": question,
                "answer": answer,
                "template_id": int(template_id) if pd.notna(template_id) else template_id,
                "question_type": question_type,
                "ecg_id": [ecg_id_val],
                "ecg_paths": [dat_path],
                "clinical_contexts": [clinical_context],
                "rationale": rationale,
            }
            out.append(sample)

        print(f"Loaded {len(out)} CoT samples for {split_name} split")
        return out

    # Build each split directly from CoT CSVs
    train_list = load_split_from_csv("train")
    val_list = load_split_from_csv("validation")
    test_list = load_split_from_csv("test")

    print("Converting to HuggingFace datasets...")
    
    # Show progress for dataset conversion
    print(f"Converting train split ({len(train_list)} samples)...")
    train_dataset = Dataset.from_list(train_list)
    print(f"Converting validation split ({len(val_list)} samples)...")
    val_dataset = Dataset.from_list(val_list)
    print(f"Converting test split ({len(test_list)} samples)...")
    test_dataset = Dataset.from_list(test_list)

    print("ECG-QA CoT dataset loading complete!")
    return train_dataset, val_dataset, test_dataset


def get_label_distribution(dataset: Dataset) -> Dict[str, int]:
    """Get the distribution of answer labels in the dataset."""
    label_counts = {}
    
    for sample in dataset:
        # Extract the actual answer label from the answer field
        answer = sample.get("answer", "")
        if isinstance(answer, str) and answer.strip():
            # For CoT datasets, the answer field contains the rationale + final answer
            # Extract the final answer after "Answer: "
            if "Answer:" in answer:
                label = answer.split("Answer:")[-1].strip().strip(".")
            else:
                # Fallback to the full answer if no "Answer:" found
                label = answer.strip()
        else:
            label = "unknown"
        
        label_counts[label] = label_counts.get(label, 0) + 1
    
    return label_counts


if __name__ == "__main__":
    # Test the loader
    print("Testing ECG-QA CoT loader...")
    
    # Test individual components
    print(f"ECG-QA CoT exists: {does_ecg_qa_cot_exist()}")
    
    try:
        train, val, test = load_ecg_qa_cot_splits()
        print("Loaded ECG-QA CoT dataset:")
        print(f"  Train: {len(train)} samples")
        print(f"  Validation: {len(val)} samples") 
        print(f"  Test: {len(test)} samples")
        
        if len(train) > 0:
            print("\nSample from training set:")
            sample = train[0]
            for key, value in sample.items():
                if isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: {value[:3]}... ({len(value)} items)")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}... ({len(value)} chars)")
                else:
                    print(f"  {key}: {value}")
                    
            # Show CoT reasoning
            if 'rationale' in sample and sample['rationale']:
                print(f"\nCoT Reasoning: {sample['rationale'][:200]}...")
                    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
