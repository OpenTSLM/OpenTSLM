# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import re
import sys
import io
import base64
from typing import Dict, Any

import matplotlib.pyplot as plt

from common_evaluator_plot import CommonEvaluatorPlot
from opentslm.time_series_datasets.TSQADataset import TSQADataset


def evaluate_tsqa(ground_truth: str, prediction: str) -> Dict[str, Any]:
    """
    Evaluate TSQA predictions against ground truth.

    - TSQA answers are in the format "(a)", "(b)", or "(c)".
    - The model might answer with a longer explanation like
      "(a) This time series has a constant volatility.", so we compare only
      the first 3 characters of the answer (e.g., "(a)").
    - If the prediction contains an "Answer:" tag, take everything after it.
    """
    # Clean up strings for comparison
    gt_clean = ground_truth.lower().strip()
    pred_clean = prediction.lower().strip()

    # Only compare the first 3 characters of the ground truth and prediction
    gt_clean, pred_clean = gt_clean[:3], pred_clean[:3]

    # Extract the actual answer from the prediction (everything after "Answer:")
    answer_match = re.search(r"answer:\s*(.+)", pred_clean, re.IGNORECASE)
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


def generate_time_series_plot(time_series) -> str:
    """
    Create a base64 PNG plot from one or more time series.

    - TSQA typically has a single 1D time series, but we handle multiple for robustness.
    - Accepts either a single 1D list/array or a list of 1D lists/arrays.
    """
    if time_series is None:
        return None

    # Ensure we have a list of series
    ts_list = list(time_series)
    # If it's a flat list of numbers, wrap it as a single series
    if ts_list and all(isinstance(x, (int, float)) for x in ts_list):
        ts_list = [ts_list]

    num_series = len(ts_list)
    fig, axes = plt.subplots(num_series, 1, figsize=(10, 4 * num_series), sharex=True)
    if num_series == 1:
        axes = [axes]

    for i, series in enumerate(ts_list):
        axes[i].plot(series, marker="o", linestyle="-", markersize=0)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f"Time Series {i + 1}")
        axes[i].set_ylabel("Value")
    axes[-1].set_xlabel("Time")

    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    img_buffer.seek(0)
    image_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return image_data


def main():
    """Main function to run TSQA evaluation with plotting."""
    if len(sys.argv) != 2:
        print("Usage: python evaluate_tsqa_plot.py <model_name>")
        print("Example: python evaluate_tsqa_plot.py openai-gpt-4o")
        sys.exit(1)

    model_name = sys.argv[1]

    dataset_classes = [TSQADataset]
    evaluation_functions = {
        "TSQADataset": evaluate_tsqa,
    }
    evaluator = CommonEvaluatorPlot()
    plot_functions = {
        "TSQADataset": generate_time_series_plot,
    }

    results_df = evaluator.evaluate_multiple_models(
        model_names=[model_name],
        dataset_classes=dataset_classes,
        evaluation_functions=evaluation_functions,
        plot_functions=plot_functions,
        max_samples=None,  # Set to None for full evaluation
        max_new_tokens=80,
    )

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    main()
