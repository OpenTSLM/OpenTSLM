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
    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str, format_sample_str: bool = False, time_series_format_function=None):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_sleepedf_cot_splits()

    def _get_answer(self, row) -> str:
        return row["rationale"]

    def _get_pre_prompt(self, _row) -> str:
        text = """
        You are given a 30-second EEG time series segment. Your task is to classify the sleep stage based on analysis of the data.

        Instructions:
        - Analyze the data objectively without presuming a particular label.
        - Reason carefully and methodically about what the signal patterns suggest regarding sleep stage.
        - Write your reasoning as a single, coherent paragraph. Do not use bullet points, lists, or section headers.
        - Only reveal the correct class at the very end.
        - Never state that you are uncertain or unable to classify the data. You must always provide a rationale and a final answer.

        
        """
        return text

    def _get_post_prompt(self, _row) -> str:
        return """Possible sleep stages are:
        Wake, Non-REM stage 1, Non-REM stage 2, Non-REM stage 3, REM sleep, Movement

        - Please now write your rationale. Make sure that your last word is the answer. You MUST end your response with "Answer: """

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
        return ["Wake", "Non-REM stage 1", "Non-REM stage 2", "Non-REM stage 3", "REM sleep", "Movement"]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["label"] = row["label"]
        sample["original_data"] = row["time_series"]
        return sample

if __name__ == "__main__":
    dataset = SleepEDFCoTQADataset(split="train", EOS_TOKEN="")
    dataset_val = SleepEDFCoTQADataset(split="validation", EOS_TOKEN="")
    dataset_test = SleepEDFCoTQADataset(split="test", EOS_TOKEN="")
    print(f"Dataset sizes: Train: {len(dataset)}, Validation: {len(dataset_val)}, Test: {len(dataset_test)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Sample answer:", sample["answer"])
        print("Sample time series text:", sample["time_series_text"] if "time_series_text" in sample else "N/A")
        print("Sample pre prompt:", sample["pre_prompt"])
        print("Sample post prompt:", sample["post_prompt"]) 