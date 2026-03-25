# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import re
import sys
from typing import Dict, Any

from common_evaluator import CommonEvaluator
from opentslm.time_series_datasets.TSQADataset import TSQADataset


def evaluate_tsqa(ground_truth: str, prediction: str) -> Dict[str, Any]:
    """
    Evaluate TSQA predictions against ground truth.
    
    Args:
        ground_truth: The correct answer
        prediction: The model's prediction
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Clean up strings for comparison
    gt_clean = ground_truth.lower().strip()
    pred_clean = prediction.lower().strip()
    
    # Only compare the first 3 characters of the ground truth and prediction,
    # because the ground truth answer is always in the format "(a)", "(b)", or "(c)",
    # but the model might answer with the full answer, e.g. "(a) This time series has a constant volatility."
    gt_clean, pred_clean = gt_clean[:3], pred_clean[:3]

    # Extract the actual answer from the prediction (everything after "Answer:")
    answer_match = re.search(r'answer:\s*(.+)', pred_clean, re.IGNORECASE)
    if answer_match:
        pred_answer = answer_match.group(1).strip()
    else:
        # If no "Answer:" found, use the entire prediction
        pred_answer = pred_clean
    
    # Calculate accuracy (exact match)
    accuracy = int(gt_clean == pred_answer)
    
    return {
        "accuracy": accuracy,
    }


def main():
    """Main function to run TSQA evaluation."""
    
    if len(sys.argv) != 2:
        print("Usage: python evaluate_tsqa.py <model_name>")
        print("Example: python evaluate_tsqa.py meta-llama/Llama-3.2-1B")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    # Define datasets to evaluate on
    dataset_classes = [TSQADataset]
    
    # Define evaluation functions
    evaluation_functions = {
        "TSQADataset": evaluate_tsqa,
    }
    
    # Initialize evaluator
    evaluator = CommonEvaluator()
    
    # Run evaluation
    results_df = evaluator.evaluate_multiple_models(
        model_names=[model_name],
        dataset_classes=dataset_classes,
        evaluation_functions=evaluation_functions,
        max_samples=None,  # Limit for faster testing, set to None for full evaluation
        max_new_tokens=40,
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    main() 