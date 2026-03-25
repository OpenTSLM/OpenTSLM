# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import re
import sys
import argparse
import io
import base64
from typing import Dict, Any

import matplotlib.pyplot as plt

from common_evaluator_plot import CommonEvaluatorPlot
from opentslm.time_series_datasets.pamap2.PAMAP2AccQADataset import PAMAP2AccQADataset
from opentslm.time_series_datasets.har_cot.HARAccQADataset import HARAccQADataset

def extract_label_from_prediction(prediction: str) -> str:
    """
    Extract the label from the model's prediction.
    - If 'Answer:' is present, take everything after the last 'Answer:'
    - Otherwise, take the last word
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
        # Take the last word
        label = pred.split()[-1] if pred.split() else ''
    # Remove trailing punctuation (e.g., period, comma)
    label = re.sub(r'[\.,;:!?]+$', '', label)
    return label.lower()


def evaluate_har_acc(ground_truth: str, prediction: str) -> Dict[str, Any]:
    """
    Evaluate HARAccQADataset predictions against ground truth.
    Extracts the label from the end of the model's output and compares to ground truth.
    """
    gt_clean = ground_truth.lower().strip()
    pred_label = extract_label_from_prediction(prediction)
    accuracy = int(gt_clean == pred_label)
    return {"accuracy": accuracy}


def generate_time_series_plot(time_series) -> str:
    """
    Create a base64 PNG plot from a list/tuple of 1D numpy arrays (e.g., [x, y, z]).
    """
    if time_series is None:
        return None
    ts_list = list(time_series)

    num_series = len(ts_list)
    fig, axes = plt.subplots(num_series, 1, figsize=(10, 4 * num_series), sharex=True)
    if num_series == 1:
        axes = [axes]

    axis_names = {0: 'X-axis', 1: 'Y-axis', 2: 'Z-axis'}
    for i, series in enumerate(ts_list):
        axes[i].plot(series, marker='o', linestyle='-', markersize=0)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f"Accelerometer - {axis_names.get(i, f'Axis {i+1}')}" )

    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    img_buffer.seek(0)
    image_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return image_data


def main():
    """Main function to run HAR evaluation."""
    if len(sys.argv) != 2:
        print("Usage: python evaluate_har_plot.py <model_name>")
        print("Example: python evaluate_har_plot.py meta-llama/Llama-3.2-1B")
        sys.exit(1)
    
    model_name = sys.argv[1]

    dataset_classes = [HARAccQADataset]
    evaluation_functions = {
        "HARAccQADataset": evaluate_har_acc,
    }
    evaluator = CommonEvaluatorPlot()
    plot_functions = {
        "HARAccQADataset": generate_time_series_plot,
    }
    results_df = evaluator.evaluate_multiple_models(
        model_names=[model_name],
        dataset_classes=dataset_classes,
        evaluation_functions=evaluation_functions,
        plot_functions=plot_functions,
        max_samples=None,  # Limit for faster testing, set to None for full evaluation,
        max_new_tokens=400,
    )
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    return results_df

if __name__ == "__main__":
    main()