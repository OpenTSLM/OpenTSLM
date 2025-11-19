#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import re
import sys
import io
import base64
from typing import Dict, Any

import matplotlib.pyplot as plt

from common_evaluator_plot import CommonEvaluatorPlot
from time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset


def extract_label_from_text(text: str) -> str:
    """Extract the label from a prediction or rationale text."""
    if text is None:
        return ""
    pred = text.strip()
    matches = list(re.finditer(r"answer:\s*", pred, re.IGNORECASE))
    if matches:
        label = pred[matches[-1].end():].strip()
    else:
        label = pred.split()[-1] if pred.split() else ""
    label = re.sub(r"[\.,;:!?]+$", "", label)
    return label.lower()


def evaluate_sleep_stage(ground_truth_text: str, prediction_text: str) -> Dict[str, Any]:
    """Evaluate SleepEDFCoTQADataset predictions against ground truth."""
    gt_label = extract_label_from_text(ground_truth_text)
    pred_label = extract_label_from_text(prediction_text)
    return {"accuracy": int(gt_label == pred_label), "gt_label": gt_label, "pred_label": pred_label}


def generate_time_series_plot(time_series) -> str:
    """Create a base64 PNG plot from the first channel (EEG) of a time series."""
    if time_series is None:
        return None
    ts_list = list(time_series)

    if len(ts_list) > 0 and hasattr(ts_list[0], "__len__"):
        eeg = ts_list[0]
    else:
        eeg = ts_list

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eeg, marker="o", linestyle="-", markersize=0)
    ax.grid(True, alpha=0.3)
    ax.set_title("EEG")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (samples)")

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
        max_samples=None,
        max_new_tokens=400,
    )

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    main()
