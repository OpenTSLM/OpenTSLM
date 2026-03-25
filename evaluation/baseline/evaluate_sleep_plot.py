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
import numpy as np

from common_evaluator_plot import CommonEvaluatorPlot
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset


def extract_label_from_text(text: str) -> str:
    """
    Extract the label from a free-form rationale or prediction text.
    - If 'Answer:' is present (case-insensitive), take everything after the last 'Answer:'
    - Otherwise, take the last word
    - Strip whitespace and trailing punctuation
    - Lowercase for comparison
    """
    if text is None:
        return ""
    pred = text.strip()
    matches = list(re.finditer(r"answer:\s*", pred, re.IGNORECASE))
    if matches:
        start = matches[-1].end()
        label = pred[start:].strip()
    else:
        label = pred.split()[-1] if pred.split() else ""
    label = re.sub(r"[\.,;:!?]+$", "", label)
    return label.lower()


def evaluate_sleep_stage(
    ground_truth_text: str, prediction_text: str
) -> Dict[str, Any]:
    """
    Evaluate SleepEDFCoTQADataset predictions against ground truth.
    For SleepEDF, the dataset's "answer" is a rationale ending with 'Answer: <label>'.
    We therefore extract the label from BOTH ground truth and prediction and compare.
    """
    gt_label = extract_label_from_text(ground_truth_text)
    pred_label = extract_label_from_text(prediction_text)
    accuracy = int(gt_label == pred_label)
    return {"accuracy": accuracy, "gt_label": gt_label, "pred_label": pred_label}


def generate_time_series_plot(time_series) -> str:
    """
    Create a base64 PNG plot from one or more time series.
    - Accepts a single 1D array/list or a collection of 1D arrays/lists.
    - If a 2D numpy array is provided, each row is treated as a separate series.
    """
    if time_series is None:
        return None
    ts_list = list(time_series)

    num_series = len(ts_list)
    fig, axes = plt.subplots(num_series, 1, figsize=(10, 4 * num_series), sharex=True)
    if num_series == 1:
        axes = [axes]

    axis_names = {0: "EEG", 1: "EOG", 2: "EMG"}
    for i, series in enumerate(ts_list):
        axes[i].plot(series, marker="o", linestyle="-", markersize=0)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f"{axis_names.get(i, f'Axis {i + 1}')}")

    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    img_buffer.seek(0)
    image_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return image_data


def main():
    """Main function to run SleepEDF evaluation with plotting."""
    if len(sys.argv) != 2:
        print("Usage: python evaluate_sleep_plot.py <model_name>")
        print("Example: python evaluate_sleep_plot.py openai-gpt-4o")
        sys.exit(1)

    model_name = sys.argv[1]

    dataset_classes = [SleepEDFCoTQADataset]
    evaluation_functions = {
        "SleepEDFCoTQADataset": evaluate_sleep_stage,
    }
    evaluator = CommonEvaluatorPlot()
    plot_functions = {
        "SleepEDFCoTQADataset": generate_time_series_plot,
    }

    results_df = evaluator.evaluate_multiple_models(
        model_names=[model_name],
        dataset_classes=dataset_classes,
        evaluation_functions=evaluation_functions,
        plot_functions=plot_functions,
        max_samples=None,  # Set to None for full evaluation
        max_new_tokens=400,
    )

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    main()
