# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import re
import sys
from typing import Dict, Any

from common_evaluator import CommonEvaluator
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset


def extract_sleep_stage_from_prediction(prediction: str) -> str:
    """
    Extract the sleep stage label from the model's prediction.
    - If 'Answer:' is present, take everything after the last 'Answer:'
    - Otherwise, take the last word/phrase
    - Strips whitespace and punctuation
    """
    pred = prediction.strip()
    # Find the last occurrence of 'Answer:' (case-insensitive)
    match = list(re.finditer(r'answer:\s*', pred, re.IGNORECASE))
    if match:
        # Take everything after the last 'Answer:'
        start = match[-1].end()
        label = pred[start:].strip()
    else:
        # Take the last word/phrase (sleep stages can be multi-word)
        words = pred.split()
        if len(words) >= 2 and words[-2].lower() in ['non-rem', 'rem']:
            # Handle multi-word sleep stages like "Non-REM stage 1", "REM sleep"
            label = ' '.join(words[-2:])
        else:
            # Single word stage
            label = words[-1] if words else ''
    # Look for "Answer: [answer]" pattern with more precise matching
    answer_match = re.search(r'Answer:\s*(.+?)(?:\.|$)', pred, re.IGNORECASE)
    if answer_match:
        label = answer_match.group(1).strip()
        # Remove trailing period if present
        if label.endswith('.'):
            label = label[:-1]
        return label.strip().lower()
    # Remove trailing punctuation (e.g., period, comma)
    label = re.sub(r'[\.,;:!?]+$', '', label)
    return label.strip().lower()


def evaluate_sleep_cot(ground_truth: str, prediction: str) -> Dict[str, Any]:
    """
    Evaluate SleepEDFCoTQADataset predictions against ground truth.
    Extracts the sleep stage label from the end of the model's output and compares to ground truth.
    """
    gt_clean = ground_truth.lower().strip()
    pred_label = extract_sleep_stage_from_prediction(prediction)
    accuracy = int(gt_clean == pred_label)
    return {"accuracy": accuracy}


def main():
    """Main function to run SleepEDF CoT evaluation."""
    if len(sys.argv) != 2:
        print("Usage: python evaluate_sleep_cot.py <model_name>")
        print("Example: python evaluate_sleep_cot.py meta-llama/Llama-3.2-1B")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    dataset_classes = [SleepEDFCoTQADataset]
    evaluation_functions = {
        "SleepEDFCoTQADataset": evaluate_sleep_cot,
    }
    
    evaluator = CommonEvaluator()
    results_df = evaluator.evaluate_multiple_models(
        model_names=[model_name],
        dataset_classes=dataset_classes,
        evaluation_functions=evaluation_functions,
        max_samples=None,  # Set to None for full evaluation
        max_new_tokens=1000,  # Sleep stage classification might need more tokens for reasoning
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    main() 