# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import zipfile
import shutil
import tempfile
from typing import Tuple

import pandas as pd
from datasets import Dataset
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Use project data root
from time_series_datasets.constants import RAW_DATA


FD_COT_DIR = os.path.join(RAW_DATA, "fault_detection")
FD_COT_ZIP_PATH = os.path.join(FD_COT_DIR, "fault_detecation_a.zip")  # archive name spelling per source
FD_COT_URL = "https://polybox.ethz.ch/index.php/s/xjx5kBLaBkesfzT"
FD_COT_DIRECT = FD_COT_URL.rstrip("/") + "/download"


def _dir_has_csvs(path: str) -> bool:
    if not os.path.exists(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if any(f.startswith("fault_detection_cot_") and f.endswith(".csv") for f in files):
            return True
    return False


def _download_file(url: str, target_path: str) -> None:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    req = Request(url, headers={"User-Agent": "OpenTSLM/1.0"})
    with urlopen(req) as resp, open(target_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    # sanity check: ensure it's a valid zip
    try:
        with zipfile.ZipFile(target_path, "r") as zf:
            _ = zf.infolist()
    except zipfile.BadZipFile:
        try:
            os.remove(target_path)
        except OSError:
            pass
        raise


def _try_download_fd_cot() -> None:
    os.makedirs(FD_COT_DIR, exist_ok=True)
    urls = [FD_COT_DIRECT, FD_COT_URL]
    for url in urls:
        try:
            _download_file(url, FD_COT_ZIP_PATH)
            if os.path.exists(FD_COT_ZIP_PATH) and os.path.getsize(FD_COT_ZIP_PATH) > 0:
                return
        except (URLError, HTTPError, zipfile.BadZipFile):
            continue
        except Exception:
            continue
    raise FileNotFoundError(
        "Failed to download Fault Detection CoT dataset. Please download manually from "
        + FD_COT_URL
        + " and place the zip at '"
        + FD_COT_ZIP_PATH
        + "'."
    )


def _extract_zip_to_target(zip_path: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            # Move all contents to target_dir (flatten any top-level folder)
            entries = os.listdir(tmpdir)
            if len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0])):
                src_root = os.path.join(tmpdir, entries[0])
            else:
                src_root = tmpdir
            for name in os.listdir(src_root):
                src = os.path.join(src_root, name)
                dst = os.path.join(target_dir, name)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        # merge
                        for r, _dnames, fnames in os.walk(src):
                            rel = os.path.relpath(r, src_root)
                            td = os.path.join(target_dir, rel)
                            os.makedirs(td, exist_ok=True)
                            for fn in fnames:
                                shutil.copy2(os.path.join(r, fn), os.path.join(td, fn))
                    else:
                        shutil.copytree(src, dst)
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)


def _ensure_dataset_available() -> None:
    if _dir_has_csvs(FD_COT_DIR):
        return
    _try_download_fd_cot()
    _extract_zip_to_target(FD_COT_ZIP_PATH, FD_COT_DIR)
    if not _dir_has_csvs(FD_COT_DIR):
        raise FileNotFoundError(
            f"Fault Detection CoT CSVs not found after extraction under '{FD_COT_DIR}'."
        )


def load_fault_detection_cot_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the Fault Detection CoT dataset splits from CSVs.

    Archive contains CSV files with columns like:
      - sample_id, question, answer, label, template_id, fault_description, prompt, rationale
    We expose these directly as HF datasets.

    Source: https://polybox.ethz.ch/index.php/s/xjx5kBLaBkesfzT (direct: .../download)
    """
    _ensure_dataset_available()

    def load_csv(name: str) -> Dataset:
        path = os.path.join(FD_COT_DIR, f"fault_detection_cot_{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split CSV: {path}")
        df = pd.read_csv(path)
        return Dataset.from_pandas(df)

    train = load_csv("train")
    val = load_csv("val")
    test = load_csv("test")
    return train, val, test


__all__ = ["load_fault_detection_cot_splits"]


