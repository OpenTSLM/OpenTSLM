# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import os
import zipfile
import requests

from opentslm.time_series_datasets.constants import RAW_DATA as RAW_DATA_PATH


# ---------------------------
# Constants
# ---------------------------

PAMAP2_URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
PAMAP2_ORG_NAME = f"{RAW_DATA_PATH}/pamap2+physical+activity+monitoring.zip"

OUTER_ZIP_NAME = "pamap2+physical+activity+monitoring.zip"
INNER_ZIP_NAME = "PAMAP2_Dataset.zip"

PAMAP2_DIR = f"{RAW_DATA_PATH}/PAMAP2_Dataset"
PROTOCOL_DIR = os.path.join(PAMAP2_DIR, "Protocol")
SUBJECT_IDS = list(range(101, 110))

# According to UCI, each .dat file contains 54 columns:
# 1 timestamp, 1 activity label, and 52 raw‐sensor features. https://archive.ics.uci.edu/ml/datasets/PAMAP2%2BPhysical%2BActivity%2BMonitoring
COLUMN_NAMES = ["timestamp", "activity"] + [f"feature_{i}" for i in range(1, 53)]

# ---------------------------
# Helper to ensure data is present
# ---------------------------


def ensure_pamap2_data(
    raw_data_path: str = RAW_DATA_PATH,
    url: str = PAMAP2_URL,
    outer_zip_name: str = OUTER_ZIP_NAME,
    inner_zip_name: str = INNER_ZIP_NAME,
):
    """
    1) Download the outer ZIP from `url` if it's not already in `raw_data_path`
    2) Extract that ZIP to drop `PAMAP2_Dataset.zip` into `raw_data_path`
    3) Extract the inner ZIP to produce the PAMAP2_Dataset/Protocol folder
    """
    # Build all the paths
    os.makedirs(raw_data_path, exist_ok=True)
    outer_zip_path = os.path.join(raw_data_path, outer_zip_name)
    inner_zip_path = os.path.join(raw_data_path, inner_zip_name)
    protocol_dir = os.path.join(raw_data_path, "PAMAP2_Dataset", "Protocol")

    # If we've already got the Protocol folder, nothing to do
    if os.path.isdir(protocol_dir):
        return

    # 1) Download outer ZIP
    if not os.path.isfile(outer_zip_path):
        print(f"Downloading PAMAP2 dataset from {url} …")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(outer_zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    # 2) Extract outer ZIP to pull out PAMAP2_Dataset.zip
    print(f"Extracting {outer_zip_path} …")
    with zipfile.ZipFile(outer_zip_path, "r") as z:
        # some UCI zips include a top-level folder,
        # but this one directly contains PAMAP2_Dataset.zip
        z.extract(inner_zip_name, path=raw_data_path)

    # 3) Extract the inner dataset ZIP
    print(f"Extracting {inner_zip_path} …")
    with zipfile.ZipFile(inner_zip_path, "r") as z:
        z.extractall(raw_data_path)
