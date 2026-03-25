# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from datasets import Dataset
from typing import List, Tuple, Literal
import numpy as np
import os

# Try to import wfdb, raise error if not available
try:
    import wfdb
except ImportError:
    raise ImportError(
        "wfdb library is required for ECG data loading. "
        "Please install it with: pip install wfdb"
    )

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import (
    load_ecg_qa_ptbxl_splits,
    load_ecg_qa_answers,
)
import pandas as pd


class ECGQADataset(QADataset):
    """
    ECG-QA Dataset for question answering with electrocardiogram data.

    This dataset combines ECG time series data from PTB-XL with
    question-answer pairs from the ECG-QA dataset.

    Requires: pip install wfdb
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
        max_samples: int = None,
        exclude_comparison: bool = False,
    ):
        """
        Initialize ECG-QA Dataset.

        Args:
            split: Dataset split to load
            EOS_TOKEN: End-of-sequence token
            format_sample_str: Whether to format samples as strings
            time_series_format_function: Function to format time series data
            max_samples: Maximum number of samples per split (for testing)
            exclude_comparison: If True, exclude comparison questions (question_type starting with "comparison_")
        """
        # Load answers mapping once
        if not hasattr(self.__class__, "answers_df"):
            print("Loading ECG-QA answers mapping...")
            self.__class__.answers_df = load_ecg_qa_answers()

        self.max_samples = max_samples
        self.exclude_comparison = exclude_comparison
        super().__init__(
            split, EOS_TOKEN, format_sample_str, time_series_format_function
        )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load the ECG-QA PTB-XL dataset splits."""
        print("Loading ECG-QA dataset splits...")
        train, val, test = load_ecg_qa_ptbxl_splits()

        # Filter out comparison questions if requested
        if self.exclude_comparison:
            print("Filtering out comparison questions...")

            def filter_comparison(dataset):
                filtered_data = []
                for sample in dataset:
                    question_type = sample.get("question_type", "")
                    if not question_type.startswith("comparison"):
                        filtered_data.append(sample)
                return Dataset.from_list(filtered_data)

            original_train_len = len(train)
            original_val_len = len(val)
            original_test_len = len(test)

            train = filter_comparison(train)
            val = filter_comparison(val)
            test = filter_comparison(test)

            print(f"Filtered out comparison questions:")
            print(
                f"  Train: {original_train_len} -> {len(train)} ({original_train_len - len(train)} removed)"
            )
            print(
                f"  Val: {original_val_len} -> {len(val)} ({original_val_len - len(val)} removed)"
            )
            print(
                f"  Test: {original_test_len} -> {len(test)} ({original_test_len - len(test)} removed)"
            )

        # Limit samples for faster testing if requested
        if self.max_samples:
            print(f"Limiting to {self.max_samples} samples per split for testing...")
            if len(train) > self.max_samples:
                train = train.select(range(self.max_samples))
            if len(val) > self.max_samples:
                val = val.select(range(self.max_samples))
            if len(test) > self.max_samples:
                test = test.select(range(self.max_samples))

        return train, val, test

    def _get_answer(self, row) -> str:
        """Extract the answer from the row."""
        # ECG-QA answers are stored as lists, typically with one answer
        if isinstance(row["answer"], list) and len(row["answer"]) > 0:
            return row["answer"][0]
        return str(row["answer"])

    def _get_pre_prompt(self, row) -> str:
        """Generate the pre-prompt explaining the task with clinical context."""
        question_type = row.get("question_type", "unknown")

        # Get clinical context if available
        clinical_contexts = row.get("clinical_contexts", [])
        clinical_context = (
            clinical_contexts[0] if clinical_contexts else "12-lead ECG recording."
        )

        base_prompt = f"""You are an expert cardiologist analyzing an ECG (electrocardiogram). 

Clinical Context: {clinical_context}

Your task is to examine the ECG signal and answer a specific medical question about it.

The ECG data shows electrical activity of the heart recorded over time. Look for patterns, 
abnormalities, and specific features that relate to the question being asked."""

        if question_type == "single-verify":
            task_specific = """

Please analyze the ECG carefully and provide a clear, definitive answer."""

        elif question_type == "single-choice":
            task_specific = """

Analyze the patterns, waves, intervals, and any abnormalities to determine the correct answer."""

        elif "comparison" in question_type:
            task_specific = """

This question requires comparison between different ECG recordings.
Look for differences, similarities, and changes between the ECGs to answer the question."""

        else:
            task_specific = """

Please analyze the ECG signal patterns carefully and provide an accurate answer."""

        return base_prompt + task_specific

    def _get_post_prompt(self, row) -> str:
        """Generate the post-prompt with possible answers and instructions."""
        # Try to get template-specific answers first
        template_id = row.get("template_id")
        possible_answers = []

        if template_id:
            possible_answers = self.get_possible_answers_for_template(template_id)

        if possible_answers:
            answers_text = ", ".join(possible_answers)
            prompt = f"""
Based on your analysis of the ECG data, select your answer from the following options:
{answers_text}

Provide a brief explanation of your reasoning, then end your response with:
Answer: <your_answer>
"""
        else:
            prompt = """
Based on your analysis of the ECG data, provide your answer.
Make sure to end your response with:
Answer: <your_answer>
"""

        return prompt.strip()

    def get_possible_answers_for_template(self, template_id: int) -> List[str]:
        """Get possible answers for a specific template ID."""
        try:
            import pandas as pd
            import ast
            from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import ECG_QA_DIR

            # Load template answers directly
            template_answers_path = os.path.join(
                ECG_QA_DIR, "ecgqa", "ptbxl", "answers_for_each_template.csv"
            )
            template_df = pd.read_csv(template_answers_path)

            # Find the row for this template_id
            template_row = template_df[template_df.template_id == template_id]
            if len(template_row) > 0:
                # Parse the string list back to actual list
                answers_str = template_row.iloc[0]["classes"]
                return ast.literal_eval(answers_str)
            else:
                print(
                    f"Warning: Template ID {template_id} not found in answers mapping"
                )
                return []

        except Exception as e:
            print(f"Error loading template answers: {e}")
            return []

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Load ECG data and convert to TextTimeSeriesPrompt format."""

        ecg_prompts = []
        ecg_paths = row.get("ecg_paths", [])

        if not ecg_paths:
            # Fallback: single ECG path
            ecg_id = row["ecg_id"][0] if row["ecg_id"] else None
            if ecg_id:
                from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import get_ptbxl_ecg_path

                ecg_path = get_ptbxl_ecg_path(ecg_id) + ".dat"
                ecg_paths = [ecg_path]

        if not ecg_paths:
            raise ValueError(f"No ECG paths found for sample. Row data: {row}")

        for i, ecg_path in enumerate(ecg_paths):
            # Load ECG data using wfdb
            base_path = ecg_path.replace(".dat", "").replace(".hea", "")

            if not os.path.exists(base_path + ".dat"):
                raise FileNotFoundError(f"ECG data file not found: {base_path}.dat")

            if not os.path.exists(base_path + ".hea"):
                raise FileNotFoundError(f"ECG header file not found: {base_path}.hea")

            try:
                # Read the ECG record
                record = wfdb.rdrecord(base_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read ECG record from {base_path}: {str(e)}"
                )

            # Get the signal data - shape is (samples, leads)
            ecg_signal = record.p_signal  # Physical signal

            if ecg_signal is None:
                raise ValueError(f"ECG signal is None for file {base_path}")

            if ecg_signal.shape[0] == 0:
                raise ValueError(
                    f"ECG signal is empty (0 samples) for file {base_path}"
                )

            # PTB-XL typically has 12 leads, sample at 500Hz for 10 seconds = 5000 samples
            # For computational efficiency, we might want to downsample

            # Take first few leads (I, II, III, aVR, aVL, aVF) which are most common
            if len(ecg_signal.shape) == 1:
                # Single lead case
                n_leads = 1
            elif len(ecg_signal.shape) == 2:
                n_leads = min(6, ecg_signal.shape[1])
                if ecg_signal.shape[1] < 6:
                    print(
                        f"Warning: ECG file {base_path} has only {ecg_signal.shape[1]} leads, expected at least 6"
                    )
            else:
                raise ValueError(
                    f"Unexpected ECG signal shape {ecg_signal.shape} for file {base_path}"
                )

            for lead_idx in range(n_leads):
                if len(ecg_signal.shape) > 1:
                    lead_signal = ecg_signal[:, lead_idx]
                else:
                    lead_signal = ecg_signal

                if len(lead_signal) == 0:
                    raise ValueError(f"Lead {lead_idx} is empty for file {base_path}")

                # Downsample from 500Hz to ~100Hz for efficiency (take every 5th sample)
                downsampled_signal = lead_signal[::5]

                if len(downsampled_signal) == 0:
                    raise ValueError(
                        f"Downsampled signal is empty for lead {lead_idx} in file {base_path}"
                    )

                # Normalize the signal
                mean_val = float(np.mean(downsampled_signal))
                std_val = float(np.std(downsampled_signal))

                if np.isnan(mean_val) or np.isnan(std_val):
                    raise ValueError(
                        f"NaN values detected in ECG signal statistics for lead {lead_idx} in file {base_path}"
                    )

                if std_val > 1e-6:  # Avoid division by zero
                    normalized_signal = (downsampled_signal - mean_val) / std_val
                else:
                    print(
                        f"Warning: Lead {lead_idx} in file {base_path} has very low std deviation ({std_val}), signal may be flat"
                    )
                    normalized_signal = downsampled_signal - mean_val

                # Verify normalized signal is valid
                if np.any(np.isnan(normalized_signal)) or np.any(
                    np.isinf(normalized_signal)
                ):
                    raise ValueError(
                        f"Invalid values (NaN/Inf) in normalized signal for lead {lead_idx} in file {base_path}"
                    )

                # Create lead name
                lead_names = [
                    "I",
                    "II",
                    "III",
                    "aVR",
                    "aVL",
                    "aVF",
                    "V1",
                    "V2",
                    "V3",
                    "V4",
                    "V5",
                    "V6",
                ]
                lead_name = (
                    lead_names[lead_idx]
                    if lead_idx < len(lead_names)
                    else f"Lead_{lead_idx}"
                )

                ecg_label = f"ECG Lead {lead_name}"
                if len(ecg_paths) > 1:
                    ecg_label += f" (Recording {i + 1})"

                ecg_label += f" - sampled at ~100Hz, normalized (mean={mean_val:.3f}, std={std_val:.3f})"

                try:
                    ecg_prompts.append(
                        TextTimeSeriesPrompt(ecg_label, normalized_signal.tolist())
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to create TextTimeSeriesPrompt for lead {lead_name} in file {base_path}: {str(e)}"
                    )

        if not ecg_prompts:
            raise RuntimeError(
                f"No ECG prompts were created for sample. ECG paths attempted: {ecg_paths}"
            )

        return ecg_prompts

    @staticmethod
    def get_labels() -> List[str]:
        """Get all possible answer labels for ECG-QA dataset."""
        # These are common answers in ECG-QA - could be loaded from answers.csv
        return [
            "yes",
            "no",
            "not sure",
            "normal",
            "abnormal",
            "borderline",
            "conduction disturbance",
            "hypertrophy",
            "ischemia",
            "infarction",
            "arrhythmia",
            "axis deviation",
            "non-specific changes",
        ]

    def _format_sample(self, row):
        # Call parent method to get the standard formatted sample
        formatted_sample = super()._format_sample(row)

        # Add template_id and question_type if they exist in the original row
        if "template_id" in row:
            formatted_sample["template_id"] = row["template_id"]
        if "question_type" in row:
            formatted_sample["question_type"] = row["question_type"]

        return formatted_sample


if __name__ == "__main__":
    # Test the dataset with limited samples
    print("Testing ECGQADataset...")

    try:
        # Test with just 10 samples per split for faster testing
        dataset = ECGQADataset(split="train", EOS_TOKEN="", max_samples=10)
        dataset_val = ECGQADataset(split="validation", EOS_TOKEN="", max_samples=10)
        dataset_test = ECGQADataset(split="test", EOS_TOKEN="", max_samples=10)

        print(
            f"Dataset sizes: Train: {len(dataset)}, Validation: {len(dataset_val)}, Test: {len(dataset_test)}"
        )

        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample keys:", sample.keys())
            print("Sample question:", sample.get("question", "N/A"))
            print("Sample answer:", sample["answer"])
            print("Sample question type:", sample.get("question_type", "N/A"))
            print("Sample ECG IDs:", sample.get("ecg_id", "N/A"))
            if "time_series_text" in sample:
                print("Time series prompts:", len(sample["time_series_text"]))
                if len(sample["time_series_text"]) > 0:
                    first_ts = sample["time_series_text"][0]
                    if hasattr(first_ts, "text"):
                        print("First time series label:", first_ts.text)
                        print("First time series length:", len(first_ts.time_series))
                    else:
                        print("Time series format:", type(first_ts))
            print(
                "Pre prompt:",
                sample["pre_prompt"][:100] + "..."
                if len(sample["pre_prompt"]) > 100
                else sample["pre_prompt"],
            )
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback

        traceback.print_exc()
