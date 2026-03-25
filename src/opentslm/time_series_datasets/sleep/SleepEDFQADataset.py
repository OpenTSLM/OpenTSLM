# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from datasets import Dataset
from typing import List, Tuple, Literal
import os
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.sleep.sleepedf_cot_loader import load_sleepedf_cot_splits
import numpy as np


class SleepEDFCoTQADataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(
            split, EOS_TOKEN, format_sample_str, time_series_format_function
        )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_sleepedf_cot_splits()

    def _get_answer(self, row) -> str:
        return row["rationale"]

    def _get_pre_prompt(self, _row) -> str:
        return ""

    def _get_post_prompt(self, _row) -> str:
        return """
You are an expert in sleep physiology and EEG interpretation. You are given only an image of a 30-second EEG segment (Fpz–Cz channel) from a de-identified, publicly available research dataset. 

Your task is to directly classify the most likely sleep stage based only on the EEG trace in the image. Do not ask for more information and do not wait for additional input — you must provide your best possible classification every time.

Steps:
1. Analyze the EEG trace in the image:
   - Identify dominant frequency bands (alpha, theta, delta, beta).
   - Look for presence or absence of sleep-specific graphoelements (sleep spindles, K-complexes, slow waves).
   - Note signal amplitude and stability.
   - Identify artifacts or irregularities.

2. Map the observed features to one of the sleep stages:
   - Wake: alpha rhythm (8-12 Hz), low-amplitude mixed-frequency, eye-blink or movement artifacts possible.
   - Non-REM stage 1: low-voltage mixed-frequency, attenuation of alpha, slow rolling eye movements.
   - Non-REM stage 2: presence of sleep spindles (11-16 Hz) and/or K-complexes, generally stable background.
   - Non-REM stage 3: high-amplitude slow waves (delta, 0.5-2 Hz, >20% of epoch).
   - REM sleep: low-amplitude mixed-frequency (theta dominant), sawtooth waves possible, no spindles or K-complexes.
   - Movement: prominent artifacts obscuring the EEG signal.

3. Always provide your answer in this exact structure:
   - **Dominant frequency/activity:** [e.g., alpha, theta, delta, mixed]
   - **Key graphoelements observed:** [spindles / K-complexes / slow waves / none]
   - **Signal characteristics:** [amplitude, stability, artifacts]
   - **Most likely sleep stage:** [Wake / Non-REM stage 1 / Non-REM stage 2 / Non-REM stage 3 / REM sleep / Movement]
   - **Confidence level:** [low / medium / high]

Rules:
- You must give a classification even if the image is unclear — in that case, state limitations but still provide your best guess.
- Never respond with a question.
- Always end with: "Answer: <sleep stage>"
"""

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        series = np.array(row["time_series"], dtype=np.float32)
        mean = float(np.mean(series))
        std = float(np.std(series))
        min_std = 1e-6
        std = max(std, min_std)
        series_norm = (series - mean) / std
        text_prompt = f"The following is the EEG time series, it has mean {mean:.4f} and std {std:.4f}:"

        return [TextTimeSeriesPrompt(text_prompt, series_norm.tolist())]

    @staticmethod
    def get_labels() -> List[str]:
        # This could be made dynamic, but for now, use the standard sleep stages
        return [
            "Wake",
            "Non-REM stage 1",
            "Non-REM stage 2",
            "Non-REM stage 3",
            "REM sleep",
            "Movement",
        ]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["label"] = row["label"]
        sample["original_data"] = row["time_series"]
        return sample


if __name__ == "__main__":
    dataset = SleepEDFCoTQADataset(split="train", EOS_TOKEN="")
    dataset_val = SleepEDFCoTQADataset(split="validation", EOS_TOKEN="")
    dataset_test = SleepEDFCoTQADataset(split="test", EOS_TOKEN="")
    print(
        f"Dataset sizes: Train: {len(dataset)}, Validation: {len(dataset_val)}, Test: {len(dataset_test)}"
    )
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Sample answer:", sample["answer"])
        print(
            "Sample time series text:",
            sample["time_series_text"] if "time_series_text" in sample else "N/A",
        )
        print("Sample pre prompt:", sample["pre_prompt"])
        print("Sample post prompt:", sample["post_prompt"])
