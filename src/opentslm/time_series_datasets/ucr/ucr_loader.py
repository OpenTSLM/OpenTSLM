# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import os
import zipfile
import requests
from typing import Literal, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# from constants import RAW_DATA_PATH

RAW_DATA_PATH = "./data"

# ---------------------------
# Constants
# ---------------------------

UCR_URL    = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
UCR_ZIP    = os.path.join(RAW_DATA_PATH, "UCRArchive_2018.zip")
UCR_DIR    = os.path.join(RAW_DATA_PATH, "UCRArchive_2018")
# Each subfolder under UCR_DIR corresponds to one dataset, e.g. "ECG5000".
# Inside each, files are named "<DatasetName>_TRAIN.tsv" and "<DatasetName>_TEST.tsv".

# ---------------------------
# Helper to ensure data
# ---------------------------

def ensure_ucr_data(
    zip_path: str = UCR_ZIP,
    extract_to: str = RAW_DATA_PATH,
    url: str      = UCR_URL
):
    """
    1) Download the UCRArchive_2018.zip if missing.
    2) Extract it to `extract_to/UCRArchive_2018`.
    """
    # Create data directory
    os.makedirs(extract_to, exist_ok=True)

    # If already extracted, skip
    if os.path.isdir(UCR_DIR):
        return

    # 1) Download ZIP if needed
    if not os.path.isfile(zip_path):
        print(f"Downloading UCR Archive from {url} …")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    # 2) Extract outer ZIP
    print(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.setpassword(b'someone')
        z.extractall(extract_to)

# ---------------------------
# Core loader
# ---------------------------

def load_ucr_dataset(
    dataset_name: str,
    raw_data_path: str = RAW_DATA_PATH
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the TRAIN and TEST TSVs for a given UCR dataset.

    Args:
        dataset_name: Name of the dataset folder (e.g. "ECG5000").
        raw_data_path: Base path where the archive is extracted.

    Returns:
        train_df, test_df: DataFrames with columns ["label", "t1", "t2", …].
    """
    ensure_ucr_data()

    base = os.path.join(raw_data_path, "UCRArchive_2018", dataset_name)
    train_path = os.path.join(base, f"{dataset_name}_TRAIN.tsv")
    test_path  = os.path.join(base, f"{dataset_name}_TEST.tsv")

    # Load using pandas; first column is label, rest are the series values
    train_df = pd.read_csv(train_path, sep="\t", header=None)
    test_df  = pd.read_csv(test_path,  sep="\t", header=None)

    # Rename columns: 0 → "label", 1...N → "t1","t2",…
    n_cols = train_df.shape[1] - 1
    col_names = ["label"] + [f"t{i}" for i in range(1, n_cols + 1)]
    train_df.columns = col_names
    test_df.columns  = col_names

    return train_df, test_df

# ---------------------------
# PyTorch Dataset + Collate
# ---------------------------

class UCRDataset(Dataset):
    """
    PyTorch Dataset for one UCR time series dataset.
    Returns (normalized series tensor, label).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        label_col: str = "label"
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols or [c for c in df.columns if c != label_col]
        self.label_col    = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        feats  = row[self.feature_cols].astype(float).values
        tensor = torch.tensor(feats, dtype=torch.float32)
        # per-sample z-normalization
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        label  = int(row[self.label_col])
        return tensor, label

def collate_fn(batch):
    """
    Stack into:
      - features: FloatTensor (batch_size, series_length)
      - labels:   LongTensor  (batch_size,)
    """
    feats, labs = zip(*batch)
    return torch.stack(feats), torch.tensor(labs, dtype=torch.long)

# ---------------------------
# DataLoader helper
# ---------------------------

def get_ucr_loader(
    dataset_name: str,
    split: Literal["train", "test", "all"] = "train",
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    raw_data_path: str = RAW_DATA_PATH
) -> DataLoader:
    """
    Returns a DataLoader for a UCR dataset.

    Args:
        dataset_name: e.g. "ECG5000".
        split:       "train", "test", or "all" to combine both.
        batch_size:  Samples per batch.
        shuffle:     Whether to shuffle (defaults to True for "train"/"all").
        raw_data_path: Base path for data.

    Returns:
        DataLoader yielding (series_tensor, label).
    """
    train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path=raw_data_path)

    if split == "train":
        df_sub = train_df
    elif split == "test":
        df_sub = test_df
    elif split == "all":
        df_sub = pd.concat([train_df, test_df], ignore_index=True)
    else:
        raise ValueError("split must be 'train', 'test', or 'all'")

    if shuffle is None:
        shuffle = split in ("train", "all")

    dataset = UCRDataset(df_sub)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    for split in ["all", "train", "test"]:
        loader = get_ucr_loader("ECG5000", split=split, batch_size=8)
        feats, labs = next(iter(loader))
        print(f"{split.capitalize()}: features {feats.shape}, labels {labs.tolist()}")