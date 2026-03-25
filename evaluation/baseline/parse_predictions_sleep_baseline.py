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
from pathlib import Path
from typing import Dict, List

# --- Inline minimal utilities (avoid importing modules that require extra packages) ---
import re


def extract_answer(text: str) -> str:
    """Extract the final answer from text by taking content after 'Answer:'
    and trimming trailing special tokens.
    """
    if "Answer: " not in text:
        return text
    answer = text.split("Answer: ")[-1].strip()
    answer = re.sub(r"<\|.*?\|>$", "", answer).strip()
    return answer


def calculate_f1_score(prediction: str, ground_truth: str):
    """Binary exact-match F1 on normalized strings (lower/strip/punct)."""
    pred_normalized = prediction.lower().strip().rstrip(".,!?;:")
    truth_normalized = ground_truth.lower().strip().rstrip(".,!?;:")
    f1 = 1.0 if pred_normalized == truth_normalized else 0.0
    return {
        "f1_score": f1,
        "precision": f1,
        "recall": f1,
        "prediction_normalized": pred_normalized,
        "ground_truth_normalized": truth_normalized,
    }


def calculate_f1_stats(data_points: List[Dict], allowed_labels=None):
    """Compute micro average and macro F1. If allowed_labels is provided, do not
    create new classes outside this set when accumulating FP counts.
    """
    if not data_points:
        return {}

    f1_scores = [point.get("f1_score", 0) for point in data_points]
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    class_predictions: Dict[str, Dict[str, int]] = {}
    if allowed_labels:
        for label in allowed_labels:
            class_predictions[label] = {"tp": 0, "fp": 0, "fn": 0}

    for point in data_points:
        gt_class = point.get("ground_truth_normalized", "")
        pred_class = point.get("prediction_normalized", "")

        if gt_class not in class_predictions:
            class_predictions[gt_class] = {"tp": 0, "fp": 0, "fn": 0}

        if pred_class == gt_class:
            class_predictions[gt_class]["tp"] += 1
        else:
            class_predictions[gt_class]["fn"] += 1
            if (allowed_labels is None) or (pred_class in (allowed_labels or set())):
                if pred_class in class_predictions:
                    class_predictions[pred_class]["fp"] += 1
                else:
                    class_predictions[pred_class] = {"tp": 0, "fp": 1, "fn": 0}

    class_f1_scores: Dict[str, Dict[str, float]] = {}
    total_f1 = 0.0
    valid_classes = 0
    for class_name, counts in class_predictions.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        class_f1_scores[class_name] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        total_f1 += f1
        valid_classes += 1

    macro_f1 = total_f1 / valid_classes if valid_classes > 0 else 0.0
    return {
        "average_f1": average_f1,
        "macro_f1": macro_f1,
        "class_f1_scores": class_f1_scores,
        "total_classes": valid_classes,
    }


def calculate_accuracy_stats(data_points: List[Dict]):
    if not data_points:
        return {}
    total = len(data_points)
    correct = sum(1 for p in data_points if p.get("accuracy", False))
    return {
        "total_samples": total,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "accuracy_percentage": (correct / total) * 100 if total else 0.0,
    }


def normalize_label(s: str) -> str:
    """Basic whitespace trim; further canonicalization is done in canonicalize_sleep_label."""
    if s is None:
        return ""
    return s.strip()


def canonicalize_sleep_label(s: str) -> str:
    """Map various aliases to the canonical SleepEDF labels (lowercased).

    Canonical target set (lowercased):
      - wake
      - non-rem stage 1
      - non-rem stage 2
      - non-rem stage 3
      - rem sleep
      - movement

    Also handle legacy/alias forms like: n1/s1/1, n2/s2/2, n3/s3/3, n4/s4/4 -> map 4 to stage 3.
    """
    if not s:
        return ""
    t = s.strip().lower()

    # Remove trailing punctuation common in generations
    while t and t[-1] in ".,;:!?":
        t = t[:-1].strip()

    # If it's an option-like artifact (e.g., "(a) wake"), keep only the label part
    # but since SleepEDF doesn't use options, just strip leading option markers if present
    if len(t) > 3 and t[0] == "(" and ")" in t[:4]:
        t = t.split(")", 1)[-1].strip()

    # Short aliases
    if t in {"w"}:
        return "wake"
    if t in {"rem", "rapid eye movement"}:
        return "rem sleep"
    if t in {"artifact", "artifacts", "movement time"}:
        return "movement"

    # Normalize common stage forms like "stage 2", "s2", "n2", "2"
    # Map to non-rem stage X; AASM stage 4 -> stage 3
    stage_map = {
        "1": "non-rem stage 1",
        "s1": "non-rem stage 1",
        "n1": "non-rem stage 1",
        "stage 1": "non-rem stage 1",
        "2": "non-rem stage 2",
        "s2": "non-rem stage 2",
        "n2": "non-rem stage 2",
        "stage 2": "non-rem stage 2",
        "3": "non-rem stage 3",
        "s3": "non-rem stage 3",
        "n3": "non-rem stage 3",
        "stage 3": "non-rem stage 3",
        "4": "non-rem stage 3",
        "s4": "non-rem stage 3",
        "n4": "non-rem stage 3",
        "stage 4": "non-rem stage 3",
    }
    if t in stage_map:
        return stage_map[t]

    # Normalize explicit non-rem phrasings
    t = t.replace("non rem", "non-rem").replace("nrem", "non-rem")
    # Handle patterns like "non-rem stage x"
    # Already canonical if it matches exactly
    canonical = {
        "wake",
        "non-rem stage 1",
        "non-rem stage 2",
        "non-rem stage 3",
        "rem sleep",
        "movement",
        # Some variants to map
        "non-rem stage 4",
    }
    if t in canonical:
        if t == "non-rem stage 4":
            return "non-rem stage 3"
        return t

    # If phrase contains key tokens, attempt heuristic mapping
    if "wake" in t:
        return "wake"
    if "rem" in t and "non-rem" not in t:
        return "rem sleep"
    if "movement" in t or "artifact" in t:
        return "movement"
    if "stage 4" in t:
        return "non-rem stage 3"
    if "stage 3" in t:
        return "non-rem stage 3"
    if "stage 2" in t or "spindle" in t or "k-complex" in t:
        return "non-rem stage 2"
    if "stage 1" in t:
        return "non-rem stage 1"

    # Fallback: return as-is (lowercased); this may be marked OOV by allowed label set
    return t


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

        ground_truth = canonicalize_sleep_label(normalize_label(gt_label))
        model_prediction = canonicalize_sleep_label(normalize_label(pred_label))
        generated = it.get("generated_answer", "")

        # Binary exact-match accuracy on normalized labels handled in calculate_f1_score, but keep explicit flag
        f1_result = calculate_f1_score(model_prediction, ground_truth)
        accuracy = f1_result["f1_score"] == 1.0

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

    # Build allowed label set from canonical SleepEDF labels (lowercased)
    ALLOWED_SLEEP_LABELS = {
        "wake",
        "non-rem stage 1",
        "non-rem stage 2",
        "non-rem stage 3",
        "rem sleep",
        "movement",
    }
    allowed_labels = ALLOWED_SLEEP_LABELS

    # F1 stats with allowed labels to prevent OOV classes from polluting per-class metrics
    f1_stats = calculate_f1_stats(data_points, allowed_labels=allowed_labels)
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
