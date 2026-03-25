# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from datasets import Dataset
from typing import List, Tuple, Literal
import os
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.ecg_qa.ecgqa_cot_loader import load_ecg_qa_cot_splits
import numpy as np

class ECGQACoTQADataset(QADataset):
    """
    ECG-QA Chain-of-Thought Dataset for question answering with electrocardiogram data.
    
    This dataset combines ECG time series data from PTB-XL with 
    question-answer pairs and chain-of-thought reasoning from the ECG-QA CoT dataset.
    
    Requires: pip install wfdb
    """
    
    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str, 
                 format_sample_str: bool = False, time_series_format_function=None,
                 max_samples: int = None, exclude_comparison: bool = False,
                 preload_processed_data: bool = True):
        """
        Initialize ECG-QA CoT Dataset.
        
        Args:
            split: Dataset split to load
            EOS_TOKEN: End-of-sequence token
            format_sample_str: Whether to format samples as strings
            time_series_format_function: Function to format time series data
            max_samples: Maximum number of samples per split (for testing)
            exclude_comparison: If True, exclude comparison questions (question_type starting with "comparison_")
            preload_processed_data: If True, preload processed ECG data for maximum speed (uses more memory). Default: True
        """
        self.max_samples = max_samples
        self.exclude_comparison = exclude_comparison
        self.preload_processed_data = preload_processed_data
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load the ECG-QA CoT dataset splits."""
        print("Loading ECG-QA CoT dataset splits...")
        train, val, test = load_ecg_qa_cot_splits()
        
        # Filter out comparison questions if requested
        if self.exclude_comparison:
            print("Filtering out comparison questions...")
            
            def filter_comparison(dataset):
                filtered_data = []
                for sample in dataset:
                    question_type = sample.get("question_type")
                    if question_type is None:
                        raise ValueError(f"Sample missing required 'question_type' field: {sample}")
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
            print(f"  Train: {original_train_len} -> {len(train)} ({original_train_len - len(train)} removed)")
            print(f"  Val: {original_val_len} -> {len(val)} ({original_val_len - len(val)} removed)")
            print(f"  Test: {original_test_len} -> {len(test)} ({original_test_len - len(test)} removed)")
        
        # Limit samples for faster testing if requested
        if self.max_samples:
            print(f"Limiting to {self.max_samples} samples per split for testing...")
            if len(train) > self.max_samples:
                train = train.select(range(self.max_samples))
            if len(val) > self.max_samples:
                val = val.select(range(self.max_samples))
            if len(test) > self.max_samples:
                test = test.select(range(self.max_samples))
        
        # Preload ECG data for better performance
        self.preload_ecg_data([train, val, test])
        
        # Optionally preload processed data for maximum performance
        if self.preload_processed_data:
            self.preload_processed_ecg_data([train, val, test])
        
        return train, val, test

    def _get_answer(self, row) -> str:
        """Get the answer from the row, which is the chain-of-thought reasoning."""
        return row.get("rationale", "No chain-of-thought reasoning available.")

    def _get_pre_prompt(self, row) -> str:
        """Generate the pre-prompt explaining the task with clinical context."""
        question_type = row.get("question_type")
        if question_type is None:
            raise ValueError(f"Sample missing required 'question_type' field: {row}")
        
        question = row.get("question")
        if question is None:
            raise ValueError(f"Sample missing required 'question' field: {row}")
        
        # Get clinical context if available
        clinical_contexts = row.get("clinical_contexts", [])
        if not clinical_contexts:
            raise ValueError(f"Sample missing required 'clinical_contexts' field: {row}")
        clinical_context = clinical_contexts[0]
        
        base_prompt = f"""You are an expert cardiologist analyzing an ECG (electrocardiogram). 

Clinical Context: {clinical_context}

Your task is to examine the ECG signal and answer the following medical question:

Question: {question}

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding the cardiac condition.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate to the question."""
        
        

        
        return base_prompt 

    def _get_post_prompt(self, row) -> str:
        """Generate the post-prompt with possible answers and instructions."""
        # Try to get template-specific answers first
        template_id = row.get("template_id")
        if template_id is None:
            raise ValueError(f"Sample missing required 'template_id' field: {row}")
        
        possible_answers = ECGQACoTQADataset.get_possible_answers_for_template(template_id)
        
        if possible_answers:
            answers_text = ", ".join(possible_answers)
            prompt = f"""
Based on your analysis of the ECG data, select your answer from the following options:
{answers_text}

- Make sure that your last word is the answer. You MUST end your response with "Answer: "
"""
        else:
            prompt = """
Based on your analysis of the ECG data, provide your answer.
Make sure that your last word is the answer. You MUST end your response with "Answer: "
"""
        
        return prompt.strip()

    # Class-level cache for template answers
    _template_answers_cache = None
    
    @classmethod
    def _load_template_answers_cache(cls):
        """Load template answers cache once."""
        if cls._template_answers_cache is None:
            try:
                import pandas as pd
                import ast
                from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import ECG_QA_DIR
                
                # Load template answers directly
                template_answers_path = os.path.join(ECG_QA_DIR, "ecgqa", "ptbxl", "answers_for_each_template.csv")
                template_df = pd.read_csv(template_answers_path)
                
                # Build cache dictionary
                cls._template_answers_cache = {}
                for _, row in template_df.iterrows():
                    template_id = row['template_id']
                    answers_str = row['classes']
                    try:
                        cls._template_answers_cache[template_id] = ast.literal_eval(answers_str)
                    except Exception as e:
                        print(f"Warning: Failed to parse answers for template {template_id}: {e}")
                        cls._template_answers_cache[template_id] = []
                        
                print(f"Loaded template answers cache with {len(cls._template_answers_cache)} templates")
                
            except Exception as e:
                print(f"Error loading template answers cache: {e}")
                cls._template_answers_cache = {}
    
    @staticmethod
    def get_possible_answers_for_template(template_id: int) -> List[str]:
        """Get possible answers for a specific template ID."""
        # Load cache if not already loaded
        ECGQACoTQADataset._load_template_answers_cache()
        
        # Return cached result
        return ECGQACoTQADataset._template_answers_cache.get(template_id, [])
    
    @staticmethod
    def get_labels() -> List[str]:
        """Get all possible answer labels from answers_for_each_template.csv."""
        try:
            import pandas as pd
            import ast
            from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import ECG_QA_DIR

            template_answers_path = os.path.join(ECG_QA_DIR, "ecgqa", "ptbxl", "answers_for_each_template.csv")
            df = pd.read_csv(template_answers_path)

            all_labels = set()
            for _, row in df.iterrows():
                classes_str = row.get("classes", "[]")
                try:
                    classes = ast.literal_eval(classes_str)
                    for c in classes:
                        if isinstance(c, str):
                            cleaned = c.strip()
                            if cleaned:
                                all_labels.add(cleaned)
                except Exception as e:
                    print(f"Warning: Failed to parse classes for template {row.get('template_id')}: {e}")
                    continue

            labels = sorted(all_labels)
            print(f"Loaded {len(labels)} unique labels from answers_for_each_template.csv")
            return labels
        except Exception as e:
            print(f"Error loading labels from CSV: {e}")
            return ["yes", "no", "not sure", "none", "normal", "abnormal"]
    
    # Class-level cache for ECG data
    _ecg_data_cache = {}
    _processed_ecg_cache = {}  # Cache for processed (downsampled + normalized) signals
    
    @classmethod
    def _load_ecg_data(cls, ecg_path: str) -> Tuple[np.ndarray, str, str]:
        """Load and cache ECG data for a given path."""
        if ecg_path not in cls._ecg_data_cache:
            try:
                import wfdb
                
                # Load ECG data using wfdb
                base_path = ecg_path.replace('.dat', '').replace('.hea', '')
                
                if not os.path.exists(base_path + '.dat'):
                    raise FileNotFoundError(f"ECG data file not found: {base_path}.dat")
                
                if not os.path.exists(base_path + '.hea'):
                    raise FileNotFoundError(f"ECG header file not found: {base_path}.hea")
                
                # Read the ECG record
                record = wfdb.rdrecord(base_path)
                
                # Get the signal data - shape is (samples, leads)
                ecg_signal = record.p_signal  # Physical signal
                
                if ecg_signal is None:
                    raise ValueError(f"ECG signal is None for file {base_path}")
                
                if ecg_signal.shape[0] == 0:
                    raise ValueError(f"ECG signal is empty (0 samples) for file {base_path}")
                
                # Determine frequency info
                if len(ecg_signal) > 1000:  # Likely 500Hz data
                    original_freq = "500Hz"
                    target_freq = "100Hz"
                else:  # Likely already 100Hz data
                    original_freq = "100Hz"
                    target_freq = "100Hz"
                
                # Cache the raw signal and frequency info
                cls._ecg_data_cache[ecg_path] = (ecg_signal, original_freq, target_freq)
                
            except Exception as e:
                raise RuntimeError(f"Failed to read ECG record from {base_path}: {str(e)}")
        
        return cls._ecg_data_cache[ecg_path]
    
    @classmethod
    def preload_ecg_data(cls, dataset_splits: List[Dataset]):
        """Preload ECG data for all samples to improve formatting performance."""
        print("Preloading ECG data for faster formatting...")
        
        # Collect all unique ECG paths
        ecg_paths = set()
        for dataset in dataset_splits:
            for sample in dataset:
                sample_ecg_paths = sample.get("ecg_paths", [])
                if sample_ecg_paths:
                    ecg_paths.update(sample_ecg_paths)
                else:
                    # Fallback: construct path from ecg_id
                    ecg_id = sample.get("ecg_id")
                    if ecg_id and isinstance(ecg_id, list) and len(ecg_id) > 0:
                        from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import get_ptbxl_ecg_path
                        ecg_path = get_ptbxl_ecg_path(ecg_id[0]) + ".dat"
                        ecg_paths.add(ecg_path)
        
        print(f"Found {len(ecg_paths)} unique ECG files to preload...")
        
        # Preload ECG data with progress bar
        from tqdm import tqdm
        for ecg_path in tqdm(ecg_paths, desc="Preloading ECG data"):
            try:
                cls._load_ecg_data(ecg_path)
            except Exception as e:
                print(f"Warning: Failed to preload ECG data from {ecg_path}: {e}")
        
        print(f"Preloaded {len(cls._ecg_data_cache)} ECG files")
    
    @classmethod
    def preload_processed_ecg_data(cls, dataset_splits: List[Dataset]):
        """Preload processed ECG data (downsampled + normalized) for all samples."""
        print("Preloading processed ECG data for maximum performance...")
        
        # Collect all unique ECG paths and lead combinations
        ecg_lead_combinations = set()
        for dataset in dataset_splits:
            for sample in dataset:
                sample_ecg_paths = sample.get("ecg_paths", [])
                if sample_ecg_paths:
                    for ecg_path in sample_ecg_paths:
                        # Load ECG data to determine number of leads
                        ecg_signal, _, _ = cls._load_ecg_data(ecg_path)
                        n_leads = ecg_signal.shape[1] if len(ecg_signal.shape) > 1 else 1
                        for lead_idx in range(n_leads):
                            ecg_lead_combinations.add((ecg_path, lead_idx))
                else:
                    # Fallback: construct path from ecg_id
                    ecg_id = sample.get("ecg_id")
                    if ecg_id and isinstance(ecg_id, list) and len(ecg_id) > 0:
                        from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import get_ptbxl_ecg_path
                        ecg_path = get_ptbxl_ecg_path(ecg_id[0]) + ".dat"
                        ecg_signal, _, _ = cls._load_ecg_data(ecg_path)
                        n_leads = ecg_signal.shape[1] if len(ecg_signal.shape) > 1 else 1
                        for lead_idx in range(n_leads):
                            ecg_lead_combinations.add((ecg_path, lead_idx))
        
        print(f"Found {len(ecg_lead_combinations)} unique ECG lead combinations to preprocess...")
        
        # Preprocess ECG leads with progress bar
        from tqdm import tqdm
        for ecg_path, lead_idx in tqdm(ecg_lead_combinations, desc="Preprocessing ECG leads"):
            try:
                cls._process_ecg_lead(ecg_path, lead_idx)
            except Exception as e:
                print(f"Warning: Failed to preprocess ECG lead {lead_idx} from {ecg_path}: {e}")
        
        print(f"Preprocessed {len(cls._processed_ecg_cache)} ECG leads")
    
    @classmethod
    def clear_caches(cls):
        """Clear all caches to free memory."""
        cls._ecg_data_cache.clear()
        cls._processed_ecg_cache.clear()
        cls._template_answers_cache = None
        print("Cleared all ECG-QA CoT dataset caches")
    
    @classmethod
    def get_cache_stats(cls) -> dict:
        """Get cache statistics for monitoring."""
        return {
            "ecg_data_cache_size": len(cls._ecg_data_cache),
            "processed_ecg_cache_size": len(cls._processed_ecg_cache),
            "template_answers_loaded": cls._template_answers_cache is not None,
            "template_answers_count": len(cls._template_answers_cache) if cls._template_answers_cache else 0
        }
    
    @classmethod
    def _process_ecg_lead(cls, ecg_path: str, lead_idx: int) -> Tuple[np.ndarray, float, float]:
        """Process and cache a single ECG lead (downsample + normalize)."""
        cache_key = f"{ecg_path}:lead_{lead_idx}"
        
        if cache_key not in cls._processed_ecg_cache:
            # Load raw ECG data
            ecg_signal, original_freq, target_freq = cls._load_ecg_data(ecg_path)
            
            # Extract lead signal
            if len(ecg_signal.shape) > 1:
                lead_signal = ecg_signal[:, lead_idx]
            else:
                lead_signal = ecg_signal
            
            if len(lead_signal) == 0:
                raise ValueError(f"Lead {lead_idx} is empty for file {ecg_path}")
            
            # Downsample if needed
            if len(lead_signal) > 1000:  # Likely 500Hz data
                downsampled_signal = lead_signal[::5]  # Downsample to 100Hz
            else:  # Already 100Hz data
                downsampled_signal = lead_signal
            
            if len(downsampled_signal) == 0:
                raise ValueError(f"Downsampled signal is empty for lead {lead_idx} in file {ecg_path}")
            
            # Normalize the signal
            mean_val = float(np.mean(downsampled_signal))
            std_val = float(np.std(downsampled_signal))
            
            if np.isnan(mean_val) or np.isnan(std_val):
                raise ValueError(f"NaN values detected in ECG signal statistics for lead {lead_idx} in file {ecg_path}")
            
            if std_val > 1e-6:  # Avoid division by zero
                normalized_signal = (downsampled_signal - mean_val) / std_val
            else:
                print(f"Warning: Lead {lead_idx} in file {ecg_path} has very low std deviation ({std_val}), signal may be flat")
                normalized_signal = downsampled_signal - mean_val
            
            # Verify normalized signal is valid
            if np.any(np.isnan(normalized_signal)) or np.any(np.isinf(normalized_signal)):
                raise ValueError(f"Invalid values (NaN/Inf) in normalized signal for lead {lead_idx} in file {ecg_path}")
            
            # Cache the processed signal and statistics
            cls._processed_ecg_cache[cache_key] = (normalized_signal, mean_val, std_val)
        
        return cls._processed_ecg_cache[cache_key]

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Load ECG data and convert to TextTimeSeriesPrompt format using cached data."""
        
        ecg_prompts = []
        ecg_paths = row.get("ecg_paths")
        if ecg_paths is None:
            raise ValueError(f"Sample missing required 'ecg_paths' field: {row}")
        
        if not ecg_paths:
            # Fallback: single ECG path
            ecg_id = row.get("ecg_id")
            if ecg_id is None:
                raise ValueError(f"Sample missing required 'ecg_id' field: {row}")
            
            if not isinstance(ecg_id, list) or len(ecg_id) == 0:
                raise ValueError(f"Sample 'ecg_id' must be a non-empty list: {ecg_id}")
            
            from opentslm.time_series_datasets.ecg_qa.ecgqa_loader import get_ptbxl_ecg_path
            ecg_path = get_ptbxl_ecg_path(ecg_id[0]) + ".dat"
            ecg_paths = [ecg_path]
        
        for i, ecg_path in enumerate(ecg_paths):
            # Load ECG data to determine number of leads
            ecg_signal, original_freq, target_freq = self._load_ecg_data(ecg_path)
            
            # Load all available leads (typically 12 for standard ECG)
            if len(ecg_signal.shape) == 1:
                # Single lead case
                n_leads = 1
            elif len(ecg_signal.shape) == 2:
                n_leads = ecg_signal.shape[1]
                if ecg_signal.shape[1] < 12:
                    print(f"Warning: ECG file {ecg_path} has only {ecg_signal.shape[1]} leads, expected 12 for standard ECG")
            else:
                raise ValueError(f"Unexpected ECG signal shape {ecg_signal.shape} for file {ecg_path}")
            
            for lead_idx in range(n_leads):
                # Use cached processed signal
                normalized_signal, mean_val, std_val = self._process_ecg_lead(ecg_path, lead_idx)
                
                # Verify we have exactly 1000 samples (10 seconds at 100Hz)
                if len(normalized_signal) != 1000:
                    print(f"Warning: Lead {lead_idx} in file {ecg_path} has {len(normalized_signal)} samples, expected 1000 for 100Hz")
                
                # Create lead name
                lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                lead_name = lead_names[lead_idx] if lead_idx < len(lead_names) else f"Lead_{lead_idx}"
                
                ecg_label = f"This is ECG Lead {lead_name}"
                if len(ecg_paths) > 1:
                    ecg_label += f" (Recording {i+1})"
                    
                ecg_label += f", it has mean {mean_val:.4f} and std {std_val:.4f}:"
                
                try:
                    ecg_prompts.append(
                        TextTimeSeriesPrompt(ecg_label, normalized_signal.tolist())
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to create TextTimeSeriesPrompt for lead {lead_name} in file {ecg_path}: {str(e)}")
        
        if not ecg_prompts:
            raise RuntimeError(f"No ECG prompts were created for sample. ECG paths attempted: {ecg_paths}")
        
        return ecg_prompts

    def _format_sample(self, row):
        # Call parent method to get the standard formatted sample
        formatted_sample = super()._format_sample(row)
        
        # Add CoT-specific fields if they exist in the original row
        if 'rationale' in row:
            formatted_sample['rationale'] = row['rationale']
        if 'cot_question_id' in row:
            formatted_sample['cot_question_id'] = row['cot_question_id']
        if 'cot_template_id' in row:
            formatted_sample['cot_template_id'] = row['cot_template_id']
        if 'cot_question_type' in row:
            formatted_sample['cot_question_type'] = row['cot_question_type']
        
        # Add original ECG-QA fields
        if 'template_id' in row:
            formatted_sample['template_id'] = row['template_id']
        if 'question_type' in row:
            formatted_sample['question_type'] = row['question_type']
        if 'question' in row:
            formatted_sample['question'] = row['question']
        
        # Add ECG data fields
        if 'ecg_id' in row:
            formatted_sample['ecg_id'] = row['ecg_id']
        if 'ecg_paths' in row:
            formatted_sample['ecg_paths'] = row['ecg_paths']
        if 'clinical_contexts' in row:
            formatted_sample['clinical_contexts'] = row['clinical_contexts']
        
        # Store the ground-truth answer according to ECG-QA and possible answers for this template
        if 'answer' in row:
            formatted_sample['correct_answer'] = row['answer']
        if 'template_id' in row and row['template_id'] is not None:
            try:
                formatted_sample['possible_answers'] = ECGQACoTQADataset.get_possible_answers_for_template(row['template_id'])
            except Exception:
                formatted_sample['possible_answers'] = []
        
        return formatted_sample

    def _format_sample_str(self, time_series_format_function, row):
        """Override to preserve template_id and other fields needed for evaluation."""
        # Call parent method to get the basic formatted sample
        formatted_sample = super()._format_sample_str(time_series_format_function, row)
        
        # Add template_id and other fields needed for evaluation
        if 'template_id' in row:
            formatted_sample['template_id'] = row['template_id']
        if 'cot_template_id' in row:
            formatted_sample['cot_template_id'] = row['cot_template_id']
        if 'question_type' in row:
            formatted_sample['question_type'] = row['question_type']
        if 'question' in row:
            formatted_sample['question'] = row['question']
        
        # Also include correct answer and possible answers to aid evaluation
        if 'answer' in row:
            formatted_sample['correct_answer'] = row['answer']
        
        return formatted_sample


if __name__ == "__main__":
    # Test the dataset with limited samples
    print("Testing ECGQACoTQADataset...")
    
    try:
        # Test with just 5 samples per split for faster testing
        dataset = ECGQACoTQADataset(split="train", EOS_TOKEN="", max_samples=5)
        dataset_val = ECGQACoTQADataset(split="validation", EOS_TOKEN="", max_samples=5)
        dataset_test = ECGQACoTQADataset(split="test", EOS_TOKEN="", max_samples=5)
        
        print(f"Dataset sizes: Train: {len(dataset)}, Validation: {len(dataset_val)}, Test: {len(dataset_test)}")
        
        # Show cache statistics
        cache_stats = ECGQACoTQADataset.get_cache_stats()
        print(f"\nCache statistics: {cache_stats}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample keys:", sample.keys())
            print("Sample question:", sample.get("question", "N/A"))
            print("Sample answer (rationale):", sample["answer"][:200] + "..." if len(sample["answer"]) > 200 else sample["answer"])
            print("Sample question type:", sample.get("question_type", "N/A"))
            print("Sample ECG IDs:", sample.get("ecg_id", "N/A"))
            if "time_series_text" in sample:
                print("Time series prompts:", len(sample["time_series_text"]))
                if len(sample["time_series_text"]) > 0:
                    first_ts = sample["time_series_text"][0]
                    if hasattr(first_ts, 'text'):
                        print("First time series label:", first_ts.text)
                        print("First time series length:", len(first_ts.time_series))
                    else:
                        print("Time series format:", type(first_ts))
            print("Pre prompt:", sample["pre_prompt"][:100] + "..." if len(sample["pre_prompt"]) > 100 else sample["pre_prompt"])
            print("Post prompt:", sample["post_prompt"])
            
            # Show CoT-specific fields
            if 'rationale' in sample:
                print("CoT Rationale:", sample['rationale'][:100] + "..." if len(sample['rationale']) > 100 else sample['rationale'])
            if 'cot_question_id' in sample:
                print("CoT Question ID:", sample['cot_question_id'])
        
        # Test performance with processed data preloading
        print("\nTesting with processed data preloading...")
        dataset_fast = ECGQACoTQADataset(split="train", EOS_TOKEN="", max_samples=3, preload_processed_data=True)
        print(f"Fast dataset size: {len(dataset_fast)}")
        
        # Show updated cache statistics
        cache_stats_after = ECGQACoTQADataset.get_cache_stats()
        print(f"Cache statistics after preloading: {cache_stats_after}")
                
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
