# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import zipfile
import shutil
import tempfile
from typing import Tuple, Dict, List, Any

import pandas as pd
from datasets import Dataset
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import ssl

# Use project data root
from time_series_datasets.constants import RAW_DATA


FD_COT_DIR = os.path.join(RAW_DATA, "fault_detection")
FD_COT_ZIP_PATH = os.path.join(FD_COT_DIR, "fault_detecation_a.zip")  # archive name spelling per source
FD_COT_URL = "https://polybox.ethz.ch/index.php/s/xjx5kBLaBkesfzT"
FD_COT_DIRECT = FD_COT_URL.rstrip("/") + "/download"

# FaultDetectionA raw time series (TS format) from UEA/Timeseries Classification site
FDA_DIR = os.path.join(RAW_DATA, "fault_detection_a")
FDA_ZIP = os.path.join(FDA_DIR, "FaultDetectionA.zip")
FDA_TRAIN_TS = os.path.join(FDA_DIR, "FaultDetectionA_TRAIN.ts")
FDA_TEST_TS = os.path.join(FDA_DIR, "FaultDetectionA_TEST.ts")
FDA_VAL_TS = os.path.join(FDA_DIR, "val.ts")  # synthetic split from tail of TRAIN
FDA_URL = "https://www.timeseriesclassification.com/aeon-toolkit/FaultDetectionA.zip"


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


# ---------------------------
# FaultDetectionA raw TS download/parse
# ---------------------------

def _download_file_simple(url: str, target_path: str) -> None:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    # Relaxed SSL for robustness
    ctx = ssl.create_default_context()
    req = Request(url, headers={"User-Agent": "OpenTSLM/1.0"})
    with urlopen(req, context=ctx) as resp, open(target_path, "wb") as out:
        shutil.copyfileobj(resp, out)


def _ensure_fault_detection_a_dataset() -> None:
    if os.path.exists(FDA_TRAIN_TS) and os.path.exists(FDA_TEST_TS):
        return
    os.makedirs(FDA_DIR, exist_ok=True)
    if not os.path.exists(FDA_ZIP):
        try:
            _download_file_simple(FDA_URL, FDA_ZIP)
        except Exception as e:
            raise RuntimeError(f"Failed to download FaultDetectionA dataset: {e}")
    try:
        with zipfile.ZipFile(FDA_ZIP, "r") as zf:
            zf.extractall(FDA_DIR)
    except Exception as e:
        raise RuntimeError(f"Failed to extract FaultDetectionA dataset: {e}")
    if not (os.path.exists(FDA_TRAIN_TS) and os.path.exists(FDA_TEST_TS)):
        raise FileNotFoundError(
            f"Missing TS files after extraction in {FDA_DIR}"
        )


def _parse_ts_line(line: str) -> Tuple[List[float] | None, float | None]:
    line = line.strip()
    if not line or line.startswith("@"):
        return None, None
    if ":" not in line:
        return None, None
    values_str, label_str = line.rsplit(":", 1)
    try:
        series = [float(x) for x in values_str.split(",") if x]
        label = float(label_str)
        return series, label
    except Exception:
        return None, None


def _load_fault_detection_a_ts(ts_path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(ts_path, "r") as f:
        for line in f:
            ts, lab = _parse_ts_line(line)
            if ts is not None and lab is not None:
                rows.append({"time_series": ts, "label": lab})
    return pd.DataFrame(rows)


def _load_fault_detection_a_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_fault_detection_a_dataset()
    combined_train = _load_fault_detection_a_ts(FDA_TRAIN_TS)
    test_df = _load_fault_detection_a_ts(FDA_TEST_TS)
    # Per dataset description: first 8184 train, last 2728 val
    train_df = combined_train.iloc[:8184].reset_index(drop=True)
    val_df = combined_train.iloc[8184:].reset_index(drop=True)
    return train_df, val_df, test_df.reset_index(drop=True)


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

    # Load raw time series splits
    ts_train_df, ts_val_df, ts_test_df = _load_fault_detection_a_splits()

    def load_and_join(name: str, ts_df: pd.DataFrame) -> Dataset:
        path = os.path.join(FD_COT_DIR, f"fault_detection_cot_{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split CSV: {path}")
        df = pd.read_csv(path)
        before = len(df)
        # Filter for non-empty rationale
        mask = (~df["rationale"].isna()) & (df["rationale"].astype(str).str.strip() != "")
        dropped = int(before - mask.sum())
        df = df.loc[mask].reset_index(drop=True)
        if dropped > 0:
            print(
                f"FaultDetectionCoT: dropped {dropped} samples without rationale from split '{name}' (kept {len(df)} of {before})."
            )

        # sample_id corresponds to index within the split's TS dataframe
        if "sample_id" not in df.columns:
            raise ValueError(f"Expected 'sample_id' column in {path}")

        # Attach time_series and sanity-check labels
        time_series_list: List[List[float]] = []
        numeric_labels: List[float] = []
        for _, row in df.iterrows():
            sid = int(row["sample_id"])
            if sid < 0 or sid >= len(ts_df):
                raise IndexError(
                    f"sample_id {sid} out of range for split '{name}' (size {len(ts_df)})"
                )
            ts_row = ts_df.iloc[sid]
            time_series_list.append(ts_row["time_series"])
            numeric_labels.append(float(ts_row["label"]))

        df = df.copy()
        df["time_series"] = time_series_list
        # Keep the original numeric label from CoT CSV if present, but also store verified
        if "label" not in df.columns:
            df["label"] = numeric_labels
        df["label_verified"] = numeric_labels

        return Dataset.from_pandas(df)

    train = load_and_join("train", ts_train_df)
    val = load_and_join("val", ts_val_df)
    test = load_and_join("test", ts_test_df)
    return train, val, test


__all__ = ["load_fault_detection_cot_splits"]


