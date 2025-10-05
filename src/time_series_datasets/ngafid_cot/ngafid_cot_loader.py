# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import json
from typing import Tuple, List, Dict, Any

import numpy as np
from datasets import Dataset
import zipfile
import tempfile
import shutil
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Use project data root
from time_series_datasets.constants import RAW_DATA


NGAFID_COT_DIR = os.path.join(RAW_DATA, "ngafid")
NGAFID_COT_ZIP_PATH = os.path.join(RAW_DATA, "ngafid", "ngafid_cot.zip")
NGAFID_COT_URL = "https://polybox.ethz.ch/index.php/s/qLDZx4BiN2JPja5"
NGAFID_COT_DIRECT_ZIP = "https://polybox.ethz.ch/index.php/s/qLDZx4BiN2JPja5/download/ngafid_cot.zip"


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _coerce_sample(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize record keys coming from ngafid_cot.zip into a consistent schema.
    
    IMPORTANT: We only use BEFORE-maintenance data for predictive maintenance tasks.
    The rationale describes what should be done based on the before-maintenance sensor readings.

    Expected possibilities (based on provided generation scripts and dataset):
      - sensor_data_before (dict[str, list[float]]) - BEFORE maintenance sensor data
      - sensor_data_after (dict[str, list[float]]) - AFTER maintenance sensor data (NOT USED)
      - rationale text in one of: rationale, gpt_prediction_rationale, answer
      - optional label/category fields
    """
    sample: Dict[str, Any] = {}

    # Sensor data: ONLY use 'sensor_data_before' for predictive maintenance
    sensors = raw.get("sensor_data_before") or {}
    # Ensure lists
    sensors = {k: (v if isinstance(v, list) else list(v)) for k, v in sensors.items()}
    sample["sensor_data"] = sensors

    # Rationale / answer text
    rationale = (
        raw.get("rationale")
        or raw.get("gpt_prediction_rationale")
        or raw.get("answer")
        or ""
    )
    # If rationale is JSON-like string, keep as text; dataset class may format further
    if not isinstance(rationale, str):
        rationale = json.dumps(rationale)
    sample["rationale"] = rationale

    # Labels / categories for reference (not required for formatting)
    sample["y_failure_within_2d"] = raw.get("y_failure_within_2d")
    sample["y_part_category"] = raw.get("y_part_category") or raw.get("target_class")
    sample["target_class_raw"] = raw.get("target_class_raw") or raw.get("label")

    # Context/meta
    sample["aircraft_context"] = (
        raw.get("aircraft_context")
        or (raw.get("flight_metadata_before", {}) or {}).get("aircraft_context")
        or "General aviation flight with continuous sensor monitoring."
    )

    # Keep ids if present
    for k in ("sample_id", "unique_sample_id", "master_index", "flight_id_before"):
        if k in raw:
            sample[k] = raw[k]

    return sample


def _dir_has_any_data(path: str) -> bool:
    if not os.path.exists(path):
        return False
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.endswith((".jsonl", ".json")):
                return True
    return False


def _download_file(url: str, target_path: str) -> None:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    req = Request(url, headers={"User-Agent": "OpenTSLM/1.0"})
    with urlopen(req) as resp, open(target_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    # Basic sanity: file should be a valid zip
    try:
        with zipfile.ZipFile(target_path, 'r') as zf:
            _ = zf.infolist()
    except zipfile.BadZipFile:
        # Not a zip (likely HTML). Remove and raise to try next URL or manual path
        try:
            os.remove(target_path)
        except OSError:
            pass
        raise


def _try_download_ngafid_cot() -> None:
    # Try direct download endpoint first (Polybox usually supports /download)
    urls = [
        NGAFID_COT_DIRECT_ZIP,
        NGAFID_COT_URL.rstrip("/") + "/download",
        NGAFID_COT_URL,
    ]
    for url in urls:
        try:
            _download_file(url, NGAFID_COT_ZIP_PATH)
            if os.path.exists(NGAFID_COT_ZIP_PATH) and os.path.getsize(NGAFID_COT_ZIP_PATH) > 0:
                return
        except (URLError, HTTPError, zipfile.BadZipFile):
            continue
        except Exception:
            continue
    raise FileNotFoundError(
        "Failed to download NGAFID CoT dataset. Please download manually from https://polybox.ethz.ch/index.php/s/qLDZx4BiN2JPja5 and place the zip at '" + NGAFID_COT_ZIP_PATH + "'."
    )


def _extract_zip_to_target(zip_path: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract to a temp dir first to handle nested folders cleanly
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            # If archive contains a top-level folder (common), descend into it
            # and move contents into target_dir
            entries = os.listdir(tmpdir)
            if len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0])):
                src_root = os.path.join(tmpdir, entries[0])
            else:
                src_root = tmpdir

            # If archive already has 'ngafid_cot' inside, flatten to target_dir
            inner_ngafid_cot = os.path.join(src_root, "ngafid_cot")
            inner_predictive = os.path.join(src_root, "predictive_dataset")
            if os.path.isdir(inner_ngafid_cot):
                src_root = inner_ngafid_cot
            elif os.path.isdir(inner_predictive):
                # Keep src_root at predictive_dataset so we can relocate later
                src_root = inner_predictive

            for name in os.listdir(src_root):
                src_path = os.path.join(src_root, name)
                dst_path = os.path.join(target_dir, name)
                if os.path.isdir(src_path):
                    if os.path.exists(dst_path):
                        # Merge directories
                        for r, dnames, fnames in os.walk(src_path):
                            rel = os.path.relpath(r, src_root)
                            td = os.path.join(target_dir, rel)
                            os.makedirs(td, exist_ok=True)
                            for fn in fnames:
                                shutil.copy2(os.path.join(r, fn), os.path.join(td, fn))
                    else:
                        shutil.copytree(src_path, dst_path)
                else:
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)


def _ensure_dataset_available() -> None:
    if _dir_has_any_data(NGAFID_COT_DIR):
        return
    # Attempt download and extract
    _try_download_ngafid_cot()
    _extract_zip_to_target(NGAFID_COT_ZIP_PATH, os.path.join(RAW_DATA, "ngafid"))
    # Post-extraction: if data landed under predictive_dataset, relocate into ngafid_cot
    pred_dir = os.path.join(RAW_DATA, "ngafid", "predictive_dataset")
    if (not _dir_has_any_data(NGAFID_COT_DIR)) and _dir_has_any_data(pred_dir):
        os.makedirs(NGAFID_COT_DIR, exist_ok=True)
        for name in os.listdir(pred_dir):
            src = os.path.join(pred_dir, name)
            dst = os.path.join(NGAFID_COT_DIR, name)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    for r, dnames, fnames in os.walk(src):
                        rel = os.path.relpath(r, pred_dir)
                        td = os.path.join(NGAFID_COT_DIR, rel)
                        os.makedirs(td, exist_ok=True)
                        for fn in fnames:
                            shutil.copy2(os.path.join(r, fn), os.path.join(td, fn))
                else:
                    shutil.copytree(src, dst)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
    # After extraction, ensure target dir exists
    if not _dir_has_any_data(NGAFID_COT_DIR):
        raise FileNotFoundError(
            f"NGAFID CoT data not found after extraction under '{NGAFID_COT_DIR}'."
        )


def _find_default_files() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load all JSON files and create stratified train/val/test splits (80/10/10) by label.
    """
    flat_candidates = [
        os.path.join(NGAFID_COT_DIR, "ngafid_cot.jsonl"),
        os.path.join(NGAFID_COT_DIR, "dataset.jsonl"),
        os.path.join(NGAFID_COT_DIR, "ngafid_cot.json"),
        os.path.join(NGAFID_COT_DIR, "dataset.json"),
    ]
    records: List[Dict[str, Any]] = []
    for path in flat_candidates:
        if os.path.exists(path) and path.endswith(".jsonl"):
            records = [_coerce_sample(r) for r in _read_jsonl(path)]
            break
        if os.path.exists(path) and path.endswith(".json"):
            with open(path, "r") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                records = [_coerce_sample(obj)]
            elif isinstance(obj, list):
                records = [_coerce_sample(r) for r in obj]
            break

    if not records:
        # Fallback: gather many per-sample JSON files inside NGAFID_COT_DIR (e.g., predictive_dataset/*.json)
        multi_json: List[Dict[str, Any]] = []
        for root, _dirs, files in os.walk(NGAFID_COT_DIR):
            for f in files:
                if f.endswith(".json"):
                    p = os.path.join(root, f)
                    try:
                        obj = json.load(open(p, "r"))
                        if isinstance(obj, dict):
                            multi_json.append(_coerce_sample(obj))
                    except Exception:
                        continue
        records = multi_json
        if not records:
            return [], [], []

    # Create stratified splits by label
    from collections import defaultdict
    label_groups = defaultdict(list)
    for record in records:
        label = record.get("target_class_raw", "unknown")
        label_groups[label].append(record)

    train, val, test = [], [], []
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    for label, group_records in label_groups.items():
        # Shuffle records for this label
        group_indices = np.arange(len(group_records))
        rng.shuffle(group_indices)
        
        # Calculate split sizes for this label
        n_total = len(group_records)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        # Split indices
        train_indices = group_indices[:n_train]
        val_indices = group_indices[n_train:n_train + n_val]
        test_indices = group_indices[n_train + n_val:]
        
        # Add to splits
        train.extend([group_records[i] for i in train_indices])
        val.extend([group_records[i] for i in val_indices])
        test.extend([group_records[i] for i in test_indices])

    # Final shuffle within each split to avoid ordering bias
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def load_ngafid_cot_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load NGAFID CoT dataset splits.

    The archive contains individual JSON files in predictive_dataset/ folder.
    We always split these files manually into train/val/test (80/10/10) with fixed seed.

    Source archive: [ngafid_cot.zip](https://polybox.ethz.ch/index.php/s/qLDZx4BiN2JPja5/download/ngafid_cot.zip)
    """
    # Ensure dataset exists locally (download+extract if needed)
    _ensure_dataset_available()

    # Always use manual splitting since archive doesn't contain split folders
    train_records, val_records, test_records = _find_default_files()

    if not (train_records and val_records and test_records):
        raise FileNotFoundError(
            f"NGAFID CoT data not found under '{NGAFID_COT_DIR}'. Please download and extract ngafid_cot.zip."
        )

    return (
        Dataset.from_list(train_records),
        Dataset.from_list(val_records),
        Dataset.from_list(test_records),
    )


__all__ = ["load_ngafid_cot_splits"]


