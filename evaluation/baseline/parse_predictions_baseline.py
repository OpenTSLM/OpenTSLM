#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Parse baseline evaluation results and compute macro-F1 from detailed per-sample outputs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from evaluation.opentslm.parse_predictions import (
    calculate_f1_score,
    calculate_f1_stats,
    calculate_accuracy_stats,
    extract_answer,
)


def extract_structured_data(obj: Dict) -> List[Dict]:
    """Extract structured per-sample data points from a detailed results JSON object."""
    items = obj.get("detailed_results", [])
    data_points: List[Dict] = []
    for it in items:
        ground_truth = it.get("target_answer", "")
        generated = it.get("generated_answer", "")
        model_prediction = extract_answer(generated)

        accuracy = model_prediction == ground_truth

        f1_result = calculate_f1_score(model_prediction, ground_truth)

        data_point = {
            "generated": generated,
            "model_prediction": model_prediction,
            "ground_truth": ground_truth,
            "accuracy": accuracy,
            "f1_score": f1_result["f1_score"],
            "precision": f1_result["precision"],
            "recall": f1_result["recall"],
            "prediction_normalized": f1_result["prediction_normalized"],
            "ground_truth_normalized": f1_result["ground_truth_normalized"],
        }
        data_points.append(data_point)
    return data_points


def main():
    ap = argparse.ArgumentParser(
        description="Compute macro-F1 from a single baseline detailed results JSON."
    )
    ap.add_argument(
        "--detailed-json",
        type=Path,
        required=True,
        help="Path to a single detailed results JSON file",
    )
    ap.add_argument(
        "--clean-out",
        type=Path,
        help="Optional path to write clean JSONL of parsed points",
    )
    args = ap.parse_args()

    with args.detailed_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    data_points = extract_structured_data(obj)

    # Accuracy stats
    accuracy_stats = calculate_accuracy_stats(data_points)
    print(f"\nAccuracy Statistics:")
    print(f"Total samples: {accuracy_stats['total_samples']}")
    print(f"Correct predictions: {accuracy_stats['correct_predictions']}")
    print(f"Incorrect predictions: {accuracy_stats['incorrect_predictions']}")
    print(f"Accuracy: {accuracy_stats['accuracy_percentage']:.2f}%")

    # F1 stats
    f1_stats = calculate_f1_stats(data_points)
    print(f"\nF1 Score Statistics:")
    print(f"Average F1 Score: {f1_stats['average_f1']:.4f}")
    print(f"Macro-F1 Score: {f1_stats['macro_f1']:.4f}")
    print(f"Total Classes: {f1_stats['total_classes']}")

    if f1_stats["class_f1_scores"]:
        print(f"\nPer-Class F1 Scores:")
        for class_name, scores in f1_stats["class_f1_scores"].items():
            print(
                f"  {class_name}: F1={scores['f1']:.4f}, P={scores['precision']:.4f}, R={scores['recall']:.4f}"
            )

    if args.clean_out:
        with args.clean_out.open("w", encoding="utf-8") as f:
            for item in data_points:
                f.write(json.dumps(item, indent=2) + "\n")
        print(f"\nData saved to {args.clean_out}")


if __name__ == "__main__":
    main()
