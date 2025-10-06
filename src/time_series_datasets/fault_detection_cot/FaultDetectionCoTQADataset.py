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
        # Prefer rationale if present; otherwise fall back to answer label
        rationale = row.get("rationale")
        if isinstance(rationale, str) and len(rationale.strip()) > 0:
            return rationale
        ans = row.get("answer")
        return str(ans) if ans is not None else ""

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
            # Fallback: use common bearing condition labels if not present
            # This keeps formatting stable even if the CSV lacks options in prompt
            options = [o for o in ["undamaged", "inner_damaged", "outer_damaged"]]

        options_block = "\n".join(f"- {opt}" for opt in options)

        text = (
            "You are presented with motor current signals from an electromechanical drive system used to monitor the condition of rolling bearings and detect damages. "
            "Possible answers:  \n"
            f"{options_block}  \n\n"
            
            "Your task is to analyze the motor current signal and determine the correct answer.\n\n"
            "Instructions:\n"
            "- Begin with a logical sequence of reasoning, see what you can observe in the signals, and then link them to the physical interpration of how the bearing fault modulates the load on the motor. Reason step by step, from the signals to the final answer, never mention the answer in the beginning!\n"
            "- NEVER mention, imply or indicate the answer in the beginning, begin with a step by step reasoning, from the signals!\n"
            "- Base your reasoning on signal characteristics such as amplitude modulation, frequency components, harmonic sidebands, and anomalies associated with bearing faults in Motor Current Signature Analysis (MCSA).  \n"
            "- Explicitly link signal patterns to the physical mechanism mechanism of how the bearing fault modulates the load on the motor.  \n"
            "- Your last sentence should include a quick summary indicating the answer, your last output should be \"Answer:"
        )
        return text

    def _get_post_prompt(self, _row) -> str:
        return "Rationale:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # The CSV does not include raw series; we construct a synthetic description placeholder.
        # For model compatibility, provide a minimal numeric series to attach with descriptive text.
        description = row.get("fault_description") or "Motor current signal segment for bearing condition assessment."
        # Provide a tiny placeholder numeric series so downstream formatting remains consistent
        series = np.asarray([0.0, 0.0, 0.0], dtype=float)
        return [TextTimeSeriesPrompt(description, series)]

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
            ("label", "numeric_label"),  # numeric label
            ("template_id", "template_id"),
            ("fault_description", "fault_description"),
        ]:
            if key_src in row:
                sample[key_dst] = row.get(key_src)
        return sample


