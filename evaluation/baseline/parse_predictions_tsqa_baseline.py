#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Parse sleep baseline evaluation results from a structured JSON file and compute
accuracy and F1 statistics. Designed for JSON files with the following shape:

{
  "model_name": "...",
  "dataset_name": "SleepEDFCoTQADataset",
  "total_samples": 930,
  "successful_inferences": 930,
  "success_rate": 1.0,
  "metrics": {"accuracy": 10.75},
  "detailed_results": [
    {
      "sample_idx": 0,
      "input_text": "...",
      "target_answer": "... Answer: Wake",
      "generated_answer": "... Answer: Wake",
      "metrics": {
        "accuracy": 1,
        "gt_label": "wake",
        "pred_label": "wake"
      }
    },
    ...
  ]
}

The script prioritizes labels under detailed_results[i]["metrics"]["gt_label"|"pred_label"],
falling back to extracting the trailing "Answer: <label>" from the target and
generated texts if labels are not provided.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from evaluation.opentslm.parse_predictions import (
    calculate_f1_score,
    calculate_f1_stats,
    calculate_accuracy_stats,
    extract_answer,
)


def normalize_label(s: str) -> str:
    """Utility to normalize a label string for comparison/printing.

    Collapses answers like "(a) ...", "a) ...", "A.", or even text that
    contains these patterns anywhere (e.g., "The answer is (a) ...") into the
    canonical form "(a)" (case-insensitive, letters a-e), so artifacts in
    generated answers are treated as just the option choice.
    """
    if s is None:
        return ""
    s = s.strip()

    # 1) Look for explicit parenthesized option anywhere: (a), (B), etc.
    m = re.search(r"\(([a-eA-E])\)", s)
    if m:
        return f"({m.group(1).lower()})"

    # 2) Look for patterns like "a)" or "B." anywhere in the string
    m = re.search(r"\b([a-eA-E])[)\.]\b", s)
    if m:
        return f"({m.group(1).lower()})"

    # 3) As a weaker fallback, detect standalone option letter bounded by word boundaries
    #    and followed by a colon or hyphen (e.g., "Answer a: ...")
    m = re.search(r"\b([a-eA-E])\s*[:\-]", s)
    if m:
        return f"({m.group(1).lower()})"

    return s


def extract_structured_data(obj: Dict) -> List[Dict]:
    """Extract structured per-sample data points from the Sleep JSON results object.

    Returns a list of dicts with keys:
      - generated
      - model_prediction
      - ground_truth
      - accuracy (bool)
      - f1_score, precision, recall
      - prediction_normalized, ground_truth_normalized
    """
    items = obj.get("detailed_results", [])
    data_points: List[Dict] = []

    for it in items:
        metrics = it.get("metrics", {}) or {}
        gt_label = metrics.get("gt_label")
        pred_label = metrics.get("pred_label")

        # Fallback to parsing the textual answers if labels are missing
        if not gt_label:
            gt_label = extract_answer(it.get("target_answer", ""))
        if not pred_label:
            pred_label = extract_answer(it.get("generated_answer", ""))

        ground_truth = normalize_label(gt_label)
        model_prediction = normalize_label(pred_label)
        generated = it.get("generated_answer", "")

        # Binary exact-match accuracy on normalized labels handled in calculate_f1_score, but keep explicit flag
        f1_result = calculate_f1_score(model_prediction, ground_truth)
        accuracy = f1_result["f1_score"] == 1.0

        # Use canonicalized labels for both values and the normalized fields used by class grouping
        data_point = {
            "generated": generated,
            "model_prediction": model_prediction,
            "ground_truth": ground_truth,
            "accuracy": accuracy,
            "f1_score": f1_result["f1_score"],
            "precision": f1_result["precision"],
            "recall": f1_result["recall"],
            "prediction_normalized": model_prediction,
            "ground_truth_normalized": ground_truth,
        }
        data_points.append(data_point)

    return data_points


def main():
    ap = argparse.ArgumentParser(
        description="Compute accuracy and F1 from a Sleep baseline results JSON (with detailed_results)."
    )
    ap.add_argument(
        "--detailed-json",
        type=Path,
        required=True,
        help="Path to a single results JSON file containing 'detailed_results'",
    )
    ap.add_argument(
        "--clean-out",
        type=Path,
        help="Optional path to write clean JSONL of parsed per-sample points",
    )
    args = ap.parse_args()

    with args.detailed_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    # Extract per-sample points
    data_points = extract_structured_data(obj)

    # Print high-level info if available
    model_name = obj.get("model_name")
    dataset_name = obj.get("dataset_name")
    total_samples = obj.get("total_samples")
    top_metrics = obj.get("metrics", {}) or {}

    if model_name or dataset_name or total_samples is not None:
        print("\nRun Metadata:")
        if model_name:
            print(f"Model: {model_name}")
        if dataset_name:
            print(f"Dataset: {dataset_name}")
        if total_samples is not None:
            print(f"Total samples (reported): {total_samples}")
        if "accuracy" in top_metrics:
            print(f"Reported accuracy: {top_metrics['accuracy']}")

    # Accuracy stats (computed from per-sample)
    accuracy_stats = calculate_accuracy_stats(data_points)
    print(f"\nAccuracy Statistics:")
    print(f"Total samples: {accuracy_stats.get('total_samples', 0)}")
    print(f"Correct predictions: {accuracy_stats.get('correct_predictions', 0)}")
    print(f"Incorrect predictions: {accuracy_stats.get('incorrect_predictions', 0)}")
    print(f"Accuracy: {accuracy_stats.get('accuracy_percentage', 0.0):.2f}%")

    # F1 stats
    f1_stats = calculate_f1_stats(data_points)
    print(f"\nF1 Score Statistics:")
    print(f"Average F1 Score: {f1_stats.get('average_f1', 0.0):.4f}")
    print(f"Macro-F1 Score: {f1_stats.get('macro_f1', 0.0):.4f}")
    print(f"Total Classes: {f1_stats.get('total_classes', 0)}")

    if f1_stats.get("class_f1_scores"):
        print(f"\nPer-Class F1 Scores:")
        for class_name, scores in f1_stats["class_f1_scores"].items():
            print(
                f"  {class_name}: F1={scores['f1']:.4f}, "
                f"P={scores['precision']:.4f}, R={scores['recall']:.4f}"
            )

    # Optional clean JSONL output
    if args.clean_out:
        with args.clean_out.open("w", encoding="utf-8") as f:
            for item in data_points:
                f.write(json.dumps(item, indent=2) + "\n")
        print(f"\nData saved to {args.clean_out}")


if __name__ == "__main__":
    main()
