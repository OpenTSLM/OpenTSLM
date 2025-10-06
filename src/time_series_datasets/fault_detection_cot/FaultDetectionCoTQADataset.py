# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import List, Tuple, Literal

import numpy as np
import torch
from datasets import Dataset

from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.fault_detection_cot.fault_detection_cot_loader import (
    load_fault_detection_cot_splits,
)


class FaultDetectionCoTQADataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_fault_detection_cot_splits()

    def _get_answer(self, row) -> str:
        # Require a non-empty rationale so the model learns to predict full reasoning
        rationale = row.get("rationale")
        if not isinstance(rationale, str) or len(rationale.strip()) == 0:
            sid = row.get("sample_id", "unknown")
            raise ValueError(
                f"FaultDetectionCoTQADataset: missing or empty 'rationale' for sample_id={sid}"
            )
        return rationale

    def _get_pre_prompt(self, row) -> str:
        # Build the desired instructional prompt using row fields
        fault_description = str(row.get("fault_description") or "")
        question = str(row.get("question") or "What is the bearing condition?")

        # Try to parse answer options from the existing prompt field if present
        def parse_options_from_prompt(p: str) -> List[str]:
            options: List[str] = []
            if not isinstance(p, str) or not p:
                return options
            for line in p.splitlines():
                line = line.strip()
                if line.startswith("- ") and len(line) > 2:
                    options.append(line[2:].strip())
            return options

        options = parse_options_from_prompt(row.get("prompt"))
        if not options:
            options = [o for o in ["undamaged", "inner_damaged", "outer_damaged"]]

        options_block = "\n".join(f"- {opt}" for opt in options)

        text = (
            "You are presented with motor current signals from an electromechanical drive system used to monitor the condition of rolling bearings and detect damages. "
            "This time series represents motor current measurements collected over a duration of 0.08 seconds (80 milliseconds), sampled at 64 kHz. "
            "The data has been segmented into 5120-point windows using a sliding window technique, preserving the original sampling rate while creating focused segments for analysis. \n\n\n"
            f"Bearing condition: {fault_description}  \n"
            f"Question: {question}  \n\n"
            "Possible answers:  \n"
            f"{options_block}  \n\n"
            "Your task is to analyze the motor current signal and determine the correct answer.\n\n"
            "Instructions:\n"
            "- Begin with a logical sequence of reasoning, see what you can observe in the signals, and then link them to the physical interpration of how the bearing fault modulates the load on the motor. Reason step by step, from the signals to the final answer, never mention the answer in the beginning!\n"
            "- NEVER mention, imply or indicate the answer in the beginning, begin with a step by step reasoning, from the signals!\n"
            "- Write your reasoning as a single coherent paragraph without lists, bullet points or section headers.\n"
            "- Base your reasoning on signal characteristics such as amplitude modulation, frequency components, harmonic sidebands, and anomalies associated with bearing faults in Motor Current Signature Analysis (MCSA).  \n"
            "- Explicitly link signal patterns to the physical mechanism mechanism of how the bearing fault modulates the load on the motor.  \n"
            "- Do not reference \"plots\" or \"visual inspection\"; only describe the inferred patterns.  \n"
            "- You are NOT allowed to mention or hint at any specific answer option, class name, or damage type until the very final sentence.  \n"
            "- Never express uncertainty. Always give deterministic reasoning.  \n"
            "- Your last sentence should include a quick summary indicating the answer, your last output should be \"Answer:"
        )
        return text

    def _get_post_prompt(self, _row) -> str:
        return "Rationale:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # Use real time series from loader
        ts = row.get("time_series")
        if ts is None:
            raise ValueError(
                f"FaultDetectionCoTQADataset: missing 'time_series' for sample_id={row.get('sample_id','unknown')}"
            )
        series = np.asarray(ts, dtype=float)
        series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
        # z-normalize per-sample for stability
        mean = float(series.mean())
        std = float(series.std())
        std = std if std > 1e-8 else 1e-8
        series_norm = (series - mean) / std
        description = (
            f"Motor current signal segment (length {len(series)}, mean {mean:.4f}, std {std:.4f})."
        )
        return [TextTimeSeriesPrompt(description, series_norm.tolist())]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        # Surface identifiers and fields to enable result-to-raw mapping later
        try:
            if "sample_id" in row:
                sample["sample_id"] = int(row["sample_id"]) if row["sample_id"] is not None else None
        except Exception:
            sample["sample_id"] = row.get("sample_id")
        # Attach semantic fields when available
        for key_src, key_dst in [
            ("question", "question"),
            ("answer", "answer_label"),  # string label
            ("label", "numeric_label"),  # numeric label (from CoT CSV)
            ("label_verified", "label_verified"),  # numeric label from raw TS
            ("template_id", "template_id"),
            ("fault_description", "fault_description"),
        ]:
            if key_src in row:
                sample[key_dst] = row.get(key_src)
        return sample


