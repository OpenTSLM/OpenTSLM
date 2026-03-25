# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from typing import Dict, Any, Callable
from common_evaluator import CommonEvaluator
from opentslm.time_series_datasets.TSQADataset import TSQADataset
from opentslm.time_series_datasets.pamap2.PAMAP2AccQADataset import PAMAP2AccQADataset
from opentslm.time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset

# Import evaluation functions
from evaluate_tsqa import evaluate_tsqa
from evaluate_pamap import evaluate_pamap_acc, evaluate_pamap_cot


def main():
    """Main function to run comprehensive evaluation across all datasets."""
    
    # Define models to evaluate
    model_names = [
        "meta-llama/Llama-3.2-1B",
        # Add more models as needed
        # "google/gemma-3n-e2b",
        # "google/gemma-3n-e2b-it",
        # "microsoft/DialoGPT-medium",
        # "gpt2",
    ]
    
    # Define datasets to evaluate on
    dataset_classes = [
        TSQADataset,
        PAMAP2AccQADataset,
        PAMAP2CoTQADataset,
    ]
    
    # Define evaluation functions
    evaluation_functions = {
        "TSQADataset": evaluate_tsqa,
        "PAMAP2AccQADataset": evaluate_pamap_acc,
        "PAMAP2CoTQADataset": evaluate_pamap_cot,
    }
    
    # Initialize evaluator
    evaluator = CommonEvaluator()
    
    # Run comprehensive evaluation
    results_df = evaluator.evaluate_multiple_models(
        model_names=model_names,
        dataset_classes=dataset_classes,
        evaluation_functions=evaluation_functions,
        max_samples=50,  # Limit for faster testing, set to None for full evaluation
    )
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Group by dataset and show average metrics
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        print(f"\n{dataset}:")
        print(f"  Number of models evaluated: {len(dataset_results)}")
        
        # Show average accuracy
        if 'accuracy' in dataset_results.columns:
            avg_accuracy = dataset_results['accuracy'].mean()
            print(f"  Average accuracy: {avg_accuracy:.1f}%")
    
    # Group by model and show average metrics
    print(f"\nBy Model:")
    for model in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model]
        print(f"\n{model}:")
        print(f"  Number of datasets evaluated: {len(model_results)}")
        
        # Show average accuracy
        if 'accuracy' in model_results.columns:
            avg_accuracy = model_results['accuracy'].mean()
            print(f"  Average accuracy: {avg_accuracy:.1f}%")
    
    return results_df


if __name__ == "__main__":
    main() 