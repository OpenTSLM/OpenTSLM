# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import json
import os
from collections import Counter


def calculate_f1_score(prediction, ground_truth):
    """Calculate F1 score for classification labels"""
    # Normalize labels for comparison (lowercase, strip whitespace and trailing punctuation)
    pred_normalized = prediction.lower().strip().rstrip(".,!?;:")
    truth_normalized = ground_truth.lower().strip().rstrip(".,!?;:")

    # For single prediction vs single ground truth, F1 is binary
    f1 = 1.0 if pred_normalized == truth_normalized else 0.0

    return {
        "f1_score": f1,
        "precision": f1,  # For single-label classification, precision = recall = f1
        "recall": f1,
        "prediction_normalized": pred_normalized,
        "ground_truth_normalized": truth_normalized,
    }


def calculate_f1_stats(data_points, allowed_labels=None):
    """Calculate both macro-F1 and average F1 (micro-F1) statistics.

    If allowed_labels is provided, predictions not in this set will:
      - contribute False Negatives to the ground-truth class, and
      - NOT count as False Positives for any (new) predicted class.
    This prevents introducing new classes into per-class/macro metrics.
    """
    if not data_points:
        return {}

    # Calculate average F1 (micro-F1) - simple average across all predictions
    f1_scores = [point.get("f1_score", 0) for point in data_points]
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    # Group by ground truth class for macro-F1
    class_predictions = {}
    if allowed_labels:
        for label in allowed_labels:
            class_predictions[label] = {"tp": 0, "fp": 0, "fn": 0}
    for point in data_points:
        gt_class = point.get("ground_truth_normalized", "")
        pred_class = point.get("prediction_normalized", "")

        if gt_class not in class_predictions:
            class_predictions[gt_class] = {"tp": 0, "fp": 0, "fn": 0}

        # True positive: prediction matches ground truth
        if pred_class == gt_class:
            class_predictions[gt_class]["tp"] += 1
        else:
            # False negative: ground truth class was not predicted
            class_predictions[gt_class]["fn"] += 1
            # False positive: predicted class that wasn't ground truth
            if (allowed_labels is None) or (pred_class in (allowed_labels or set())):
                if pred_class in class_predictions:
                    class_predictions[pred_class]["fp"] += 1
                else:
                    class_predictions[pred_class] = {"tp": 0, "fp": 1, "fn": 0}

    # Calculate F1 per class
    class_f1_scores = {}
    total_f1 = 0
    valid_classes = 0

    for class_name, counts in class_predictions.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
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

    # Calculate macro-F1 (average across all classes)
    macro_f1 = total_f1 / valid_classes if valid_classes > 0 else 0

    return {
        "average_f1": average_f1,
        "macro_f1": macro_f1,
        "class_f1_scores": class_f1_scores,
        "total_classes": valid_classes,
    }


# Path to your JSONL file
file_path = "evaluation_results_openai-gpt-4o_tsqadataset.json"

# Check if file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

if os.path.getsize(file_path) == 0:
    print(f"File is empty: {file_path}")
    exit(1)

# Counters
total = 0
correct = 0
data_points = []
labels = ["(a)", "(b)", "(c)"]
label_to_idx = {l: i for i, l in enumerate(labels)}
confusion = [[0, 0, 0] for _ in range(3)]  # rows: gold, cols: pred
support = {l: 0 for l in labels}

# Read and process the file
with open(file_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            print(f"Skipping empty line at {line_num}")
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON decode error on line {line_num}: {e}")
            continue

        generated_raw = entry.get("generated", "").strip()
        gold_raw = entry.get("gold", "").strip()

        # Use only the first three characters for comparison (e.g., "(a)")
        generated = generated_raw[:3]
        gold = gold_raw[:3]

        total += 1
        is_correct = generated == gold
        if is_correct:
            correct += 1
        else:
            # Only print incorrect predictions
            print(
                f"Line {line_num} - Generated: {generated_raw} -> {generated}, Gold: {gold_raw} -> {gold}"
            )

        # Calculate F1 score for this prediction
        f1_result = calculate_f1_score(generated, gold)

        data_point = {
            "generated": generated,
            "gold": gold,
            "accuracy": is_correct,
            "f1_score": f1_result["f1_score"],
            "precision": f1_result["precision"],
            "recall": f1_result["recall"],
            "prediction_normalized": f1_result["prediction_normalized"],
            "ground_truth_normalized": f1_result["ground_truth_normalized"],
        }
        data_points.append(data_point)

        # Update confusion matrix and supports when labels are recognized
        if gold in label_to_idx and generated in label_to_idx:
            gi = label_to_idx[gold]
            pi = label_to_idx[generated]
            confusion[gi][pi] += 1
            support[gold] += 1

# Compute and print accuracy
if total == 0:
    print("No valid entries found.")
else:
    accuracy = correct / total
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})")

    # Calculate and display F1 statistics
    allowed_labels = {point.get("ground_truth_normalized", "") for point in data_points}
    f1_stats = calculate_f1_stats(data_points, allowed_labels=allowed_labels)

    print(f"\nF1 Score Statistics:")
    print(f"Average F1 Score: {f1_stats['average_f1']:.4f}")
    print(f"Macro-F1 Score: {f1_stats['macro_f1']:.4f}")
    print(f"Total Classes: {f1_stats['total_classes']}")

    # Display per-class F1 scores
    if f1_stats["class_f1_scores"]:
        print(f"\nPer-Class F1 Scores:")
        for class_name, scores in f1_stats["class_f1_scores"].items():
            print(
                f"  {class_name}: F1={scores['f1']:.4f}, P={scores['precision']:.4f}, R={scores['recall']:.4f}"
            )

    # Print class supports
    print("\nClass support (gold counts):")
    for l in labels:
        print(f"  {l}: {support.get(l, 0)}")

    # Print confusion matrix
    print("\nConfusion matrix (rows=gold, cols=pred):")
    header = "       " + "  ".join(labels)
    print(header)
    for i, l in enumerate(labels):
        row = "  ".join(str(x) for x in confusion[i])
        print(f"  {l}  {row}")
