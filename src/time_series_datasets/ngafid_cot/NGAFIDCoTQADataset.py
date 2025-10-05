# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import List, Tuple, Literal

import os
import sys
import torch
from datasets import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.ngafid_cot.ngafid_cot_loader import load_ngafid_cot_splits


SENSOR_LABELS_ORDERED = [
    "engine_temperature",  # CHT combined proxy
    "engine_performance",  # EGT combined proxy
    "oil_parameters",      # OilT/OilP
    "fuel_system",         # Fuel qty/flow
    "electrical",          # Volt/Amp
    "engine_condition",    # RPM
]


class NGAFIDCoTQADataset(QADataset):
    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str, format_sample_str: bool = False, time_series_format_function=None):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_ngafid_cot_splits()

    def _get_answer(self, row) -> str:
        # Parse the JSON rationale and format it properly
        rationale = row["rationale"]
        try:
            import json
            if isinstance(rationale, str):
                parsed = json.loads(rationale)
                # Format as structured text
                formatted = f"State Summary: {parsed.get('state_summary', '')}\n\n"
                formatted += f"Recommended Action: {parsed.get('recommended_action', '')}\n\n"
                formatted += f"Expected Outcome: {parsed.get('expected_outcome', '')}"
                return formatted
            else:
                return str(rationale)
        except (json.JSONDecodeError, TypeError):
            # Fallback to raw text if not JSON
            return str(rationale)

    def _get_pre_prompt(self, row) -> str:
        context = row.get("aircraft_context", "General aviation flight with continuous sensor monitoring.")
        text = f"""
You are an aircraft maintenance forecaster analyzing flight sensor data to predict maintenance needs. Consider the aircraft context: {context}

Your task is to perform predictive maintenance analysis based on the sensor readings. You will see sensor data from a flight that occurred before maintenance was performed. Your job is to:

1. Analyze the sensor patterns to identify potential issues or anomalies
2. Recommend specific maintenance actions to address the problems
3. Predict the expected outcome if those maintenance actions are taken

Instructions:
- Examine the sensor data patterns carefully for signs of degradation, anomalies, or impending failures
- Think step-by-step about what the observed patterns suggest regarding engine health and system performance
- Provide your analysis in three structured sections:

1. State Summary: Describe what you observe in the sensor data patterns and what issues they indicate
2. Recommended Action: Suggest specific maintenance actions to address the identified problems
3. Expected Outcome: Predict what improvements you expect after the recommended maintenance is performed

Focus on predictive maintenance - identifying problems before they become critical failures.

Provide your analysis in the three structured sections as described above.
"""
        return text

    def _get_post_prompt(self, _row) -> str:
        return "Rationale:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        sensor_data = row["sensor_data"]

        # Convert to a fixed ordered list for the model, normalizing each series
        prompts: List[TextTimeSeriesPrompt] = []
        for name in SENSOR_LABELS_ORDERED:
            if name not in sensor_data:
                # Skip missing series; downstream formatting should tolerate varying counts
                continue
            series = torch.tensor(sensor_data[name], dtype=torch.float32)

            if torch.isnan(series).any() or torch.isinf(series).any():
                # Handle NaN/Inf values by creating a descriptive message
                text_prompt = f"The following is the {name.replace('_', ' ')} sensor stream. The data contains NaN (not available) values:"
                # Create a simple placeholder series with zeros
                placeholder_series = [0.0] * len(series)
                prompts.append(TextTimeSeriesPrompt(text_prompt, placeholder_series))
                continue

            mean = series.mean()
            std = series.std()
            std = torch.clamp(std, min=1e-6)
            series_norm = (series - mean) / std

            if torch.isnan(series_norm).any() or torch.isinf(series_norm).any():
                # Handle NaN/Inf after normalization
                text_prompt = f"The following is the {name.replace('_', ' ')} sensor stream. The normalized data contains NaN (not available) values:"
                placeholder_series = [0.0] * len(series)
                prompts.append(TextTimeSeriesPrompt(text_prompt, placeholder_series))
                continue

            text_prompt = f"The following is the {name.replace('_', ' ')} sensor stream, it has mean {float(mean):.4f} and std {float(std):.4f}:"
            prompts.append(TextTimeSeriesPrompt(text_prompt, series_norm.tolist()))

        if not prompts:
            raise ValueError(f"No sensor data found in NGAFID CoT sample {row.get('sample_id', 'unknown')}")

        return prompts

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        # Keep raw for analysis/debug
        sample["sensor_data_keys"] = list(row["sensor_data"].keys())
        sample["y_failure_within_2d"] = row.get("y_failure_within_2d")
        sample["y_part_category"] = row.get("y_part_category")
        sample["target_class_raw"] = row.get("target_class_raw")
        return sample


if __name__ == "__main__":
    ds_train = NGAFIDCoTQADataset(split="train", EOS_TOKEN="")
    ds_val = NGAFIDCoTQADataset(split="validation", EOS_TOKEN="")
    ds_test = NGAFIDCoTQADataset(split="test", EOS_TOKEN="")
    print(f"Dataset sizes: Train={len(ds_train)}, Val={len(ds_val)}, Test={len(ds_test)}")
    if len(ds_train) > 0:
        s = ds_train[0]
        print("Sample keys:", s.keys())
        print("Answer preview:", (s["answer"][:200] + "...") if len(s["answer"]) > 200 else s["answer"]) 


