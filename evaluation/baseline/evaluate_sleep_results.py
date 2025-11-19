#!/usr/bin/env python3
"""
Evaluate sleep inference results from a CSV file.
Reuses evaluation logic from parse_sleep_cot_data.py.
"""
import argparse
import re
from typing import Dict, Any, List

import pandas as pd

# Standard sleep stage labels (from SleepEDFCoTQADataset)
FALLBACK_LABELS = [
    "Wake",
    "Non-REM stage 1",
    "Non-REM stage 2",
    "Non-REM stage 3",
    "REM sleep",
    "Movement",
]
SUPPORTED_LABELS: List[str] = []


def _canonicalize_label(text: str) -> tuple:
    """Return canonical label with stage 4 merged into stage 3.

    Handles both short labels (W, N1, N2, N3, N4, REM) and long labels
    (Wake, Non-REM stage 1, etc.).

    - Case-insensitive
    - Trims whitespace and trailing punctuation
    - Merges stage 4 into stage 3
    - Returns (canonical_label_str, is_supported_bool)
    """
    if text is None:
        return "", False

    cleaned = str(text).strip()
    # Remove any end-of-text tokens and trailing punctuation
    cleaned = re.sub(r'<\|.*?\|>|<eos>$', '', cleaned).strip()
    cleaned = re.sub(r'[\.,;:!?]+$', '', cleaned).strip()

    lowered = cleaned.lower()

    # Map short labels to canonical forms first
    short_label_map = {
        "w": "Wake",
        "wake": "Wake",
        "awake": "Wake",
        "n1": "Non-REM stage 1",
        "n2": "Non-REM stage 2",
        "n3": "Non-REM stage 3",
        "n4": "Non-REM stage 3",  # Merge stage 4 into stage 3
        "rem": "REM sleep",
        "r": "REM sleep",
        "mov": "Movement",
        "mt": "Movement",
        "movement": "Movement",
    }

    if lowered in short_label_map:
        canonical = short_label_map[lowered]
    # Normalize common variants and merge stage 4 into stage 3
    elif "non-rem" in lowered or "nrem" in lowered:
        lowered = lowered.replace("nrem", "non-rem")
        lowered = lowered.replace("non rem", "non-rem")

        # Map stage 4 -> stage 3
        if "stage 4" in lowered:
            canonical = "Non-REM stage 3"
        elif "stage 3" in lowered:
            canonical = "Non-REM stage 3"
        elif "stage 2" in lowered:
            canonical = "Non-REM stage 2"
        elif "stage 1" in lowered:
            canonical = "Non-REM stage 1"
        else:
            canonical = cleaned
    elif "rem" in lowered and "sleep" in lowered:
        canonical = "REM sleep"
    else:
        label_set = SUPPORTED_LABELS if SUPPORTED_LABELS else FALLBACK_LABELS
        maybe = next((lab for lab in label_set if lab.lower() == lowered), "")
        canonical = maybe if maybe else cleaned

    label_set = SUPPORTED_LABELS if SUPPORTED_LABELS else FALLBACK_LABELS
    is_supported = canonical in label_set
    return canonical if canonical else cleaned, is_supported


def extract_answer(text: str) -> str:
    """Extract the final answer from text.

    Handles formats like:
    - "Answer: Wake" -> "Wake"
    - "Answer: Non-REM stage 2" -> "Non-REM stage 2"
    - "Answer: N2" -> "N2"
    """
    if text is None:
        return ""
    text = str(text)

    if "Answer: " not in text:
        return text.strip()

    # Take the last "Answer: " and get what follows
    answer = text.split("Answer: ")[-1].strip()

    # Take only the first line (in case model continues generating)
    answer = answer.split("\n")[0].strip()

    # Remove any end-of-text tokens
    answer = re.sub(r'<\|.*?\|>|<eos>$', '', answer).strip()
    # Remove trailing punctuation
    answer = re.sub(r'[\.,;:!?]+$', '', answer).strip()

    return answer


def calculate_f1_score(prediction: str, ground_truth: str) -> Dict[str, Any]:
    """Calculate F1 score for single-label classification with supported labels."""
    pred_canon, pred_supported = _canonicalize_label(prediction)
    truth_canon, truth_supported = _canonicalize_label(ground_truth)

    f1 = 1.0 if pred_canon == truth_canon else 0.0

    return {
        'f1_score': f1,
        'precision': f1,
        'recall': f1,
        'prediction_normalized': pred_canon.lower().strip(),
        'ground_truth_normalized': truth_canon.lower().strip(),
        'prediction_supported': pred_supported,
        'ground_truth_supported': truth_supported,
    }


def calculate_f1_stats(data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate both macro-F1 and average F1 (micro-F1) statistics."""
    if not data_points:
        return {}

    f1_scores = [point.get("f1_score", 0) for point in data_points]
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    labels_to_use = SUPPORTED_LABELS if SUPPORTED_LABELS else FALLBACK_LABELS
    supported_lower = {label.lower(): label for label in labels_to_use}
    class_predictions = {lab.lower(): {"tp": 0, "fp": 0, "fn": 0} for lab in labels_to_use}

    for point in data_points:
        gt_class = point.get("ground_truth_normalized", "")
        pred_class = point.get("prediction_normalized", "")
        pred_supported = point.get("prediction_supported", False)

        if gt_class not in class_predictions:
            continue

        if pred_class == gt_class:
            class_predictions[gt_class]["tp"] += 1
        else:
            class_predictions[gt_class]["fn"] += 1
            if pred_supported and pred_class in class_predictions:
                class_predictions[pred_class]["fp"] += 1

    class_f1_scores = {}
    total_f1 = 0
    valid_classes = 0

    for class_name, counts in class_predictions.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        pretty_name = supported_lower.get(class_name, class_name)
        class_f1_scores[pretty_name] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        total_f1 += f1
        valid_classes += 1

    macro_f1 = total_f1 / valid_classes if valid_classes > 0 else 0

    return {
        "average_f1": average_f1,
        "macro_f1": macro_f1,
        "class_f1_scores": class_f1_scores,
        "total_classes": valid_classes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate sleep inference results from CSV")
    parser.add_argument("csv_path", type=str, help="Path to inference results CSV")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-sample details")
    args = parser.parse_args()

    global SUPPORTED_LABELS

    # Load CSV
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} samples from {args.csv_path}\n")

    # First pass: discover labels from ground truth
    discovered_labels = set()
    for _, row in df.iterrows():
        gt_raw = extract_answer(str(row["target_answer"]))
        gt_canon, _ = _canonicalize_label(gt_raw)
        if gt_canon:
            discovered_labels.add(gt_canon)

    SUPPORTED_LABELS = list(discovered_labels)
    print(f"Discovered {len(SUPPORTED_LABELS)} labels from ground truth:")
    for label in sorted(SUPPORTED_LABELS):
        print(f"  - {label}")
    print()

    # Second pass: evaluate each sample
    data_points = []
    for idx, row in df.iterrows():
        gt_raw = extract_answer(str(row["target_answer"]))
        pred_raw = extract_answer(str(row["generated_answer"]))

        pred_canon, pred_supported = _canonicalize_label(pred_raw)
        gt_canon, gt_supported = _canonicalize_label(gt_raw)

        accuracy = (pred_canon == gt_canon) and gt_supported
        f1_result = calculate_f1_score(pred_raw, gt_raw)

        data_point = {
            "accuracy": accuracy,
            "f1_score": f1_result['f1_score'],
            "precision": f1_result['precision'],
            "recall": f1_result['recall'],
            "prediction_normalized": f1_result['prediction_normalized'],
            "ground_truth_normalized": f1_result['ground_truth_normalized'],
            "prediction_supported": f1_result['prediction_supported'],
            "ground_truth_supported": f1_result['ground_truth_supported'],
        }
        data_points.append(data_point)

        if args.verbose:
            status = "✓" if accuracy else "✗"
            print(f"[{idx}] {status} GT: {gt_canon} | Pred: {pred_canon}")

    # Calculate statistics
    total = len(data_points)
    correct = sum(1 for p in data_points if p.get("accuracy", False))
    accuracy_pct = (correct / total) * 100 if total > 0 else 0

    f1_stats = calculate_f1_stats(data_points)

    # Print results
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy_pct:.2f}%")
    print(f"\nAverage F1 Score: {f1_stats.get('average_f1', 0):.4f}")
    print(f"Macro-F1 Score: {f1_stats.get('macro_f1', 0):.4f}")

    # Per-class statistics
    class_f1_scores = f1_stats.get("class_f1_scores", {})
    if class_f1_scores:
        print(f"\nPer-Class Statistics:")
        for class_name, scores in sorted(class_f1_scores.items()):
            print(f"  {class_name}:")
            print(f"    F1: {scores['f1']:.4f}, Precision: {scores['precision']:.4f}, Recall: {scores['recall']:.4f}")
            print(f"    TP: {scores['tp']}, FP: {scores['fp']}, FN: {scores['fn']}")


if __name__ == "__main__":
    main()
