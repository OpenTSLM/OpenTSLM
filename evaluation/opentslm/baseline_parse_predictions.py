#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Parser for converting baseline JSON files to clean format."""

import json
import re
import sys
import os
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Import the dataset class to get labels
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

# Get the supported labels from the dataset class
SUPPORTED_LABELS = HARCoTQADataset.get_labels()


def calculate_f1_score(prediction, ground_truth):
    """Calculate F1 score for classification labels using token overlap (like SQuAD)."""

    # Normalize labels
    pred_normalized = prediction.lower().strip().rstrip(".,!?;:")
    truth_normalized = ground_truth.lower().strip().rstrip(".,!?;:")

    pred_tokens = pred_normalized.split()
    truth_tokens = truth_normalized.split()

    if not pred_tokens and not truth_tokens:
        return {
            "f1_score": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "prediction_normalized": pred_normalized,
            "ground_truth_normalized": truth_normalized,
        }
    if not pred_tokens or not truth_tokens:
        return {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "prediction_normalized": pred_normalized,
            "ground_truth_normalized": truth_normalized,
        }

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(truth_tokens) if truth_tokens else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )

    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "prediction_normalized": pred_normalized,
        "ground_truth_normalized": truth_normalized,
    }


def calculate_f1_stats(data_points, allowed_labels=None):
    """Calculate macro-F1 across classes from confusion counts."""

    if not data_points:
        return {}

    class_predictions = {}
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

    class_f1_scores = {}
    total_f1 = 0
    valid_classes = 0

    for class_name, counts in class_predictions.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
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

    macro_f1 = total_f1 / valid_classes if valid_classes > 0 else 0

    return {
        "macro_f1": macro_f1,
        "class_f1_scores": class_f1_scores,
        "total_classes": valid_classes,
    }


def calculate_accuracy_stats(data_points):
    """Calculate accuracy statistics from data points"""
    if not data_points:
        return {}

    total = len(data_points)
    correct = sum(1 for point in data_points if point.get("accuracy", False))
    accuracy_percentage = (correct / total) * 100 if total > 0 else 0

    return {
        "total_samples": total,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "accuracy_percentage": accuracy_percentage,
    }


def parse_baseline_json(input_file, output_file=None):
    """Parse baseline JSON file and extract structured data."""
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}.clean.jsonl")

    print(f"Parsing {input_file}")
    print(f"Output will be saved to {output_file}")

    print(f"Using {len(SUPPORTED_LABELS)} labels from HARCoTQADataset:")
    for label in sorted(SUPPORTED_LABELS):
        print(f"  - {label}")

    extracted_data = extract_structured_data(input_file)

    allowed_labels = set(SUPPORTED_LABELS)

    answer_count = sum(
        1 for point in extracted_data if "Answer: " in point.get("generated", "")
    )

    excluded_count = 0
    for point in extracted_data:
        prediction_label = point.get("prediction_normalized", "")
        is_valid_prediction = prediction_label in allowed_labels
        point["excluded"] = not is_valid_prediction
        if not is_valid_prediction:
            excluded_count += 1

    if extracted_data:
        print(f"Extracted {len(extracted_data)} data points")
        print(f"Predictions containing 'Answer:': {answer_count}/{len(extracted_data)}")
        if excluded_count > 0:
            print(
                f"Excluded {excluded_count} predictions not in the label set from metrics"
            )

        accuracy_stats = calculate_accuracy_stats(extracted_data)
        print(f"\nAccuracy Statistics:")
        print(f"Total samples: {accuracy_stats['total_samples']}")
        print(f"Correct predictions: {accuracy_stats['correct_predictions']}")
        print(f"Incorrect predictions: {accuracy_stats['incorrect_predictions']}")
        print(f"Accuracy: {accuracy_stats['accuracy_percentage']:.2f}%")

        f1_stats = calculate_f1_stats(extracted_data, allowed_labels=allowed_labels)
        print(f"\nF1 Score Statistics:")
        print(f"Macro-F1 Score: {f1_stats['macro_f1']:.4f}")
        print(f"Total Classes: {f1_stats['total_classes']}")

        if f1_stats["class_f1_scores"]:
            print(f"\nPer-Class F1 Scores:")
            for class_name, scores in f1_stats["class_f1_scores"].items():
                print(
                    f"  {class_name}: "
                    f"F1={scores['f1']:.4f}, "
                    f"P={scores['precision']:.4f}, "
                    f"R={scores['recall']:.4f}, "
                    f"TP={scores['tp']}, FP={scores['fp']}, FN={scores['fn']}"
                )

        with open(output_file, "w", encoding="utf-8") as f:
            for item in extracted_data:
                f.write(json.dumps(item, indent=2) + "\n")

        print(f"\nData saved to {output_file}")
        return extracted_data
    else:
        print("No data could be extracted from the file.")
        return []


def extract_structured_data(input_file):
    """Extract structured data from baseline JSON file"""
    data_points = []

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        detailed_results = data.get("detailed_results", [])

        for result in tqdm(detailed_results, desc="Processing results"):
            try:
                sample_idx = result.get("sample_idx", 0)
                generated_answer = result.get("generated_answer", "")
                target_answer = result.get("target_answer", "")

                model_prediction = extract_answer(generated_answer).replace("<eos>", "")
                ground_truth = extract_answer(target_answer).replace("<eos>", "")

                accuracy = (
                    model_prediction.strip().lower() == ground_truth.strip().lower()
                )

                f1_result = calculate_f1_score(model_prediction, ground_truth)

                data_point = {
                    "sample_idx": sample_idx,
                    "generated": generated_answer,
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
            except Exception as e:
                print(
                    f"Error processing sample {result.get('sample_idx', 'unknown')}: {e}"
                )
                continue

    return data_points


def extract_answer(text):
    """Extract the final answer from text"""
    if "Answer: " not in text:
        return text.strip()
    answer = text.split("Answer: ")[-1].strip()
    answer = re.sub(r"<\|.*?\|>$", "", answer).strip()
    return answer


if __name__ == "__main__":
    # Example usage with the baseline JSON file
    input_file = "evaluation_results_meta-llama-llama-3-2-3b_harcotqadataset.json"
    output_file = "out.jsonl"

    parse_baseline_json(input_file, output_file)
