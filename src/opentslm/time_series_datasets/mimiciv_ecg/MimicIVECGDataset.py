# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from datasets import Dataset
from typing import List, Tuple, Literal, Optional, Dict, Callable
from functools import partial
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
from opentslm.time_series_datasets.mimiciv_ecg.mimiciv_ecg_loader import (
    load_mimiciv_ecg_splits,
    get_mimiciv_ecg_path,
)
import torch


# ECG-QA lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
ECG_QA_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Common MimicIV lead orders (may vary)
# Some MimicIV datasets have aVR and aVF swapped
MIMICIV_ALTERNATIVE_ORDER = ["I", "II", "III", "aVF", "aVL", "aVR", "V1", "V2", "V3", "V4", "V5", "V6"]


class MimicIVECGDataset(QADataset):
    """
    MimicIV-ECG Dataset for question answering with electrocardiogram data.
    
    This dataset loads ECG time series data from MimicIV-ECG and allows
    custom prompts to be provided for evaluation.
    
    Requires: pip install wfdb
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
        max_samples: int = None,
        ecg_data_dir: Optional[str] = None,
            custom_prompts: Optional[List[str]] = None,  # Required for MimicIV-ECG
        pre_prompt_template: Optional[str] = None,
        post_prompt_template: Optional[str] = None,
        fix_lead_order: bool = True,
        use_custom_prompt_directly: bool = False,
    ):
        """
        Initialize MimicIV-ECG Dataset.
        
        Args:
            split: Dataset split to load
            EOS_TOKEN: End-of-sequence token
            format_sample_str: Whether to format samples as strings
            time_series_format_function: Function to format time series data
            max_samples: Maximum number of samples per split (for testing)
            ecg_data_dir: Directory containing ECG data files (.dat/.hea)
            custom_prompts: Optional list of custom prompts to use for each sample.
                          If provided, these will be used instead of dataset questions.
                          Should be a list with one prompt per sample, or a function
                          that takes (sample_idx, sample) and returns a prompt.
            pre_prompt_template: Optional template for pre-prompt. Can use {question} placeholder.
            post_prompt_template: Optional template for post-prompt.
            fix_lead_order: If True, reorder leads to match ECG-QA order (I, II, III, aVR, aVL, aVF, V1-V6)
            use_custom_prompt_directly: If True, use custom prompt as-is without default wrapper. 
                                      When True, pre_prompt_template and post_prompt_template are ignored.
        """
        self.max_samples = max_samples
        self.ecg_data_dir = ecg_data_dir
        self.custom_prompts = custom_prompts
        self.pre_prompt_template = pre_prompt_template
        self.post_prompt_template = post_prompt_template
        self.fix_lead_order = fix_lead_order
        self.use_custom_prompt_directly = use_custom_prompt_directly
        
        # Override parent initialization to handle NaN samples
        self.EOS_TOKEN = EOS_TOKEN
        if not hasattr(self.__class__, "loaded"):
            train, val, test = self._load_splits()

            format_function = partial(self._format_sample_str, time_series_format_function) if format_sample_str else self._format_sample
           
            from tqdm import tqdm
            
            # Format samples and skip those with NaN values
            def format_with_skip(split_name, dataset):
                formatted_samples = []
                skipped_count = 0
                for sample in tqdm(dataset, desc=f"Formatting {split_name} samples"):
                    try:
                        formatted = format_function(sample)
                        formatted_samples.append(formatted)
                    except (ValueError, RuntimeError) as e:
                        if "NaN" in str(e) or "Inf" in str(e) or "SAMPLE_SKIP" in str(e):
                            skipped_count += 1
                            continue
                        raise
                if skipped_count > 0:
                    print(f"⚠️  Excluded {skipped_count} samples with NaN/Inf values from {split_name} split")
                return formatted_samples
            
            print("Formatting training samples...")
            self.__class__._train_dataset = format_with_skip("training", train)
            
            print("Formatting validation samples...")
            self.__class__._validation_dataset = format_with_skip("validation", val)
            
            print("Formatting test samples...")
            self.__class__._test_dataset = format_with_skip("test", test)

            self.__class__.loaded = True

        match split:
            case "train":
                self.dataset = self.__class__._train_dataset
            case "validation":
                self.dataset = self.__class__._validation_dataset
            case "test":
                self.dataset = self.__class__._test_dataset
            case _:
                raise RuntimeError(
                    "Split must be a literal of 'train', 'test', or 'validation'"
                )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load the MimicIV-ECG dataset splits."""
        print("Loading MimicIV-ECG dataset splits...")
        train, val, test = load_mimiciv_ecg_splits(
            ecg_data_dir=self.ecg_data_dir,
            max_samples=self.max_samples
        )
        return train, val, test

    def _get_answer(self, row) -> str:
        """Extract the answer from the row."""
        # For evaluation, we might not have ground truth answers
        # Return empty string or placeholder
        if isinstance(row.get("answer"), list) and len(row.get("answer", [])) > 0:
            return str(row["answer"][0])
        return str(row.get("answer", ""))

    def _get_pre_prompt(self, row) -> str:
        """Generate the pre-prompt with custom prompt."""
        # Custom prompts must be provided - no fallback to dataset questions
        if self.custom_prompts is None:
            raise ValueError(
                "custom_prompts must be provided. MimicIV-ECG dataset requires explicit prompts."
            )
        
        if callable(self.custom_prompts):
            # Function: call with row data
            sample_idx = row.get("sample_id", 0)
            question = self.custom_prompts(sample_idx, row)
        elif isinstance(self.custom_prompts, list):
            # List: use sample_id if available, otherwise hash row for consistency
            sample_idx = row.get("sample_id", None)
            if sample_idx is not None:
                prompt_idx = sample_idx % len(self.custom_prompts)
            else:
                # Use hash of ecg_id for consistent mapping
                ecg_ids = row.get("ecg_id", [])
                if ecg_ids:
                    prompt_idx = hash(str(ecg_ids[0])) % len(self.custom_prompts)
                else:
                    prompt_idx = 0
            question = self.custom_prompts[prompt_idx]
        else:
            # Single string: use for all samples
            question = str(self.custom_prompts)
        
        # If use_custom_prompt_directly is True, return the prompt as-is
        if self.use_custom_prompt_directly:
            return question.strip()
        
        # Use template if provided
        if self.pre_prompt_template:
            pre_prompt = self.pre_prompt_template.format(question=question)
        else:
            # Default pre-prompt - match ECGQACoTQADataset format
            # Note: MimicIV doesn't have clinical_context, so we omit that line
            pre_prompt = f"""You are an expert cardiologist analyzing an ECG (electrocardiogram). 

Your task is to examine the ECG signal and answer the following medical question:

Question: {question}

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding the cardiac condition.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate to the question."""
        
        return pre_prompt

    def _get_post_prompt(self, row) -> str:
        """Generate the post-prompt with instructions."""
        # If use_custom_prompt_directly is True, return empty post-prompt
        if self.use_custom_prompt_directly:
            return ""
        
        if self.post_prompt_template:
            return self.post_prompt_template.strip()
        
        # Default post-prompt - match ECGQACoTQADataset format
        return """Based on your analysis of the ECG data, provide your answer.
Make sure that your last word is the answer. You MUST end your response with "Answer: "
"""

    def _reorder_leads_to_ecg_qa_order(
        self, ecg_signal: np.ndarray, record: wfdb.Record
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Reorder ECG leads to match ECG-QA order: I, II, III, aVR, aVL, aVF, V1-V6.
        
        Args:
            ecg_signal: ECG signal array (samples, leads)
            record: wfdb Record object with lead names
        
        Returns:
            Tuple of (reordered_signal, lead_names)
        """
        if not self.fix_lead_order:
            # Return original order
            lead_names = record.sig_name if hasattr(record, 'sig_name') else [
                f"Lead_{i}" for i in range(ecg_signal.shape[1])
            ]
            return ecg_signal, lead_names
        
        # Get lead names from record
        if hasattr(record, 'sig_name') and record.sig_name:
            record_lead_names = [name.strip().upper() for name in record.sig_name]
        else:
            # Fallback: assume standard order
            record_lead_names = ECG_QA_LEAD_ORDER[:ecg_signal.shape[1]]
        
        # Create mapping from record order to ECG-QA order
        reordered_signal = np.zeros_like(ecg_signal)
        reordered_lead_names = []
        
        for target_idx, target_lead in enumerate(ECG_QA_LEAD_ORDER):
            if target_idx >= ecg_signal.shape[1]:
                break
            
            # Try to find the lead in the record
            found = False
            for record_idx, record_lead in enumerate(record_lead_names):
                # Normalize lead names for comparison
                normalized_record = record_lead.replace("AVR", "AVR").replace("AVL", "AVL").replace("AVF", "AVF")
                normalized_target = target_lead.replace("aVR", "AVR").replace("aVL", "AVL").replace("aVF", "AVF")
                
                if normalized_record == normalized_target or record_lead == target_lead:
                    reordered_signal[:, target_idx] = ecg_signal[:, record_idx]
                    reordered_lead_names.append(target_lead)
                    found = True
                    break
            
            if not found:
                # If lead not found, try alternative order (aVR/aVF swapped)
                if target_lead == "aVR":
                    # Try aVF position
                    for record_idx, record_lead in enumerate(record_lead_names):
                        if "AVF" in record_lead.upper():
                            reordered_signal[:, target_idx] = ecg_signal[:, record_idx]
                            reordered_lead_names.append(target_lead)
                            found = True
                            break
                elif target_lead == "aVF":
                    # Try aVR position
                    for record_idx, record_lead in enumerate(record_lead_names):
                        if "AVR" in record_lead.upper():
                            reordered_signal[:, target_idx] = ecg_signal[:, record_idx]
                            reordered_lead_names.append(target_lead)
                            found = True
                            break
                
                if not found:
                    # If still not found, use the lead at this position
                    if target_idx < len(record_lead_names):
                        reordered_signal[:, target_idx] = ecg_signal[:, target_idx]
                        reordered_lead_names.append(ECG_QA_LEAD_ORDER[target_idx])
                    else:
                        # Pad with zeros if we run out of leads
                        reordered_lead_names.append(ECG_QA_LEAD_ORDER[target_idx])
        
        return reordered_signal, reordered_lead_names

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Load ECG data and convert to TextTimeSeriesPrompt format."""
        ecg_prompts = []
        ecg_paths = row.get("ecg_paths", [])

        if not ecg_paths:
            # Fallback: try to construct path from ecg_id
            ecg_id = row.get("ecg_id", [])
            if ecg_id and len(ecg_id) > 0:
                ecg_path = get_mimiciv_ecg_path(ecg_id[0], self.ecg_data_dir) + ".dat"
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

            # Reorder leads to match ECG-QA order if requested
            if self.fix_lead_order:
                ecg_signal, lead_names = self._reorder_leads_to_ecg_qa_order(
                    ecg_signal, record
                )
            else:
                lead_names = (
                    record.sig_name
                    if hasattr(record, "sig_name") and record.sig_name
                    else [f"Lead_{j}" for j in range(ecg_signal.shape[1])]
                )

            # Process each lead
            n_leads = min(12, ecg_signal.shape[1])  # Use up to 12 leads

            for lead_idx in range(n_leads):
                if len(ecg_signal.shape) > 1:
                    lead_signal = ecg_signal[:, lead_idx]
                else:
                    lead_signal = ecg_signal

                if len(lead_signal) == 0:
                    raise ValueError(f"Lead {lead_idx} is empty for file {base_path}")

                # Downsample from 500Hz to ~100Hz for efficiency (take every 5th sample)
                if len(lead_signal) > 1000:
                    downsampled_signal = lead_signal[::5]
                else:
                    downsampled_signal = lead_signal

                if len(downsampled_signal) == 0:
                    raise ValueError(
                        f"Downsampled signal is empty for lead {lead_idx} in file {base_path}"
                    )

                # Check for NaN/Inf values in the signal before processing
                if np.any(np.isnan(downsampled_signal)) or np.any(np.isinf(downsampled_signal)):
                    # Skip this sample - return empty list to signal it should be skipped
                    raise ValueError(
                        f"NaN/Inf values detected in ECG signal for lead {lead_idx} "
                        f"in file {base_path}. Sample will be excluded."
                    )
                
                # Normalize the signal
                mean_val = float(np.mean(downsampled_signal))
                std_val = float(np.std(downsampled_signal))

                if np.isnan(mean_val) or np.isnan(std_val):
                    raise ValueError(
                        f"NaN values detected in ECG signal statistics for lead {lead_idx} "
                        f"in file {base_path}. Sample will be excluded."
                    )

                if std_val > 1e-6:  # Avoid division by zero
                    normalized_signal = (downsampled_signal - mean_val) / std_val
                else:
                    print(
                        f"Warning: Lead {lead_idx} in file {base_path} has very low std deviation "
                        f"({std_val}), signal may be flat"
                    )
                    normalized_signal = downsampled_signal - mean_val

                # Verify normalized signal is valid
                if np.any(np.isnan(normalized_signal)) or np.any(
                    np.isinf(normalized_signal)
                ):
                    raise ValueError(
                        f"Invalid values (NaN/Inf) in normalized signal for lead {lead_idx} "
                        f"in file {base_path}"
                    )

                # Create lead name - match exact format from ECGQACoTQADataset
                lead_names_list = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                lead_name = lead_names_list[lead_idx] if lead_idx < len(lead_names_list) else f"Lead_{lead_idx}"
                
                ecg_label = f"This is ECG Lead {lead_name}"
                if len(ecg_paths) > 1:
                    ecg_label += f" (Recording {i+1})"
                    
                ecg_label += f", it has mean {mean_val:.4f} and std {std_val:.4f}:"

                try:
                    ecg_prompts.append(
                        TextTimeSeriesPrompt(ecg_label, normalized_signal.tolist())
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to create TextTimeSeriesPrompt for lead {lead_name} "
                        f"in file {base_path}: {str(e)}"
                    )

        if not ecg_prompts:
            raise RuntimeError(
                f"No ECG prompts were created for sample. ECG paths attempted: {ecg_paths}"
            )

        return ecg_prompts

    def _format_sample(self, row):
        # Call parent method to get the standard formatted sample
        formatted_sample = super()._format_sample(row)

        # Add metadata if available
        if "ecg_id" in row:
            formatted_sample["ecg_id"] = row["ecg_id"]
        if "template_id" in row:
            formatted_sample["template_id"] = row.get("template_id")
        if "question_type" in row:
            formatted_sample["question_type"] = row.get("question_type")
        if "sample_id" in row:
            formatted_sample["sample_id"] = row.get("sample_id")

        return formatted_sample


if __name__ == "__main__":
    # Test the dataset
    print("Testing MimicIVECGDataset...")

    try:
        # Test with just 5 samples per split for faster testing
        dataset = MimicIVECGDataset(
            split="test", EOS_TOKEN="", max_samples=5
        )

        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample keys:", sample.keys())
            print("Sample question:", sample.get("pre_prompt", "N/A")[:200])
            print("Sample ECG IDs:", sample.get("ecg_id", "N/A"))
            if "time_series_text" in sample:
                print("Time series prompts:", len(sample["time_series_text"]))
                if len(sample["time_series_text"]) > 0:
                    first_ts = sample["time_series_text"][0]
                    if hasattr(first_ts, "text"):
                        print("First time series label:", first_ts.text)
                        print("First time series length:", len(first_ts.time_series))

    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback

        traceback.print_exc()
