# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
CoT datasets whose *training target* is the short label / CSV `answer` field only.

Separate classes keep QADataset class-level format caches independent from full CoT runs.
Prompts and time series match the parent CoT datasets; only `_get_answer` changes.
"""

from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset


class HARCoTQADatasetAnswerOnly(HARCoTQADataset):
    """HAR CoT samples; supervise `label` (activity class) instead of full rationale."""

    def _get_answer(self, row) -> str:
        return row["label"]


class SleepEDFCoTQADatasetAnswerOnly(SleepEDFCoTQADataset):
    """Sleep CoT samples; supervise `label` (sleep stage) instead of full rationale."""

    def _get_answer(self, row) -> str:
        return row["label"]


class ECGQACoTQADatasetAnswerOnly(ECGQACoTQADataset):
    """ECG-QA CoT samples; supervise CSV `answer` (short phrase) instead of rationale."""

    def _get_answer(self, row) -> str:
        a = row.get("answer")
        if a is None or not str(a).strip():
            raise ValueError("ECG-QA CoT sample missing non-empty `answer` field.")
        return str(a).strip()
