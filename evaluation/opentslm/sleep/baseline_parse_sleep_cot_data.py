#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Parser for converting baseline sleep COT JSON files to clean format."""

import json
import re
from pathlib import Path
from tqdm import tqdm

from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset

# We'll determine supported labels dynamically from the actual ground truth data
# Start with the dataset class labels as a fallback
FALLBACK_LABELS = SleepEDFCoTQADataset.get_labels()
SUPPORTED_LABELS = []  # Will be populated dynamically


def _canonicalize_label(text):
    """Return canonical label with stage 4 merged into stage 3.

    - Case-insensitive
    - Trims whitespace and trailing period
    - Merges "non-rem stage 4" -> "Non-REM stage 3"
    - Returns (canonical_label_str, is_supported_bool)
    """
    if text is None:
        return "", False

    cleaned = str(text).strip()
    # Remove any end-of-text tokens and trailing period
    cleaned = re.sub(r"<\|.*?\|>|<eos>$", "", cleaned).strip()
    cleaned = re.sub(r"\.$", "", cleaned).strip()

    lowered = cleaned.lower()

    # Normalize common variants and merge stage 4 into stage 3
    if "non-rem" in lowered or "nrem" in lowered:
        # unify spacing/hyphenation
        lowered = lowered.replace("nrem", "non-rem")
        lowered = lowered.replace("non rem", "non-rem")

    # Map stage 4 -> stage 3
    if "non-rem" in lowered and "stage 4" in lowered:
        canonical = "Non-REM stage 3"
    elif "non-rem" in lowered and "stage 3" in lowered:
        canonical = "Non-REM stage 3"
    elif "non-rem" in lowered and "stage 2" in lowered:
        canonical = "Non-REM stage 2"
    elif "non-rem" in lowered and "stage 1" in lowered:
        canonical = "Non-REM stage 1"
    elif "rem" in lowered and "sleep" in lowered:
        canonical = "REM sleep"
    elif lowered in {"wake", "awake"}:
        canonical = "Wake"
    elif "movement" in lowered or lowered == "mov" or lowered == "mt":
        canonical = "Movement"
    else:
        # If it exactly matches a supported label ignoring case, keep it
        # Use fallback labels if supported labels haven't been determined yet
        label_set = SUPPORTED_LABELS if SUPPORTED_LABELS else FALLBACK_LABELS
        maybe = next((lab for lab in label_set if lab.lower() == lowered), "")
        canonical = maybe if maybe else cleaned

    # Use fallback labels if supported labels haven't been determined yet
    label_set = SUPPORTED_LABELS if SUPPORTED_LABELS else FALLBACK_LABELS
    is_supported = canonical in label_set
    return canonical if canonical else cleaned, is_supported


def calculate_f1_score(prediction, ground_truth):
    """Calculate F1 score for single-label classification with supported labels.

    - Merges Non-REM stage 4 into stage 3
    - Only counts predictions within SUPPORTED_LABELS; unsupported predictions yield F1=0
    """
    pred_canon, pred_supported = _canonicalize_label(prediction)
    truth_canon, truth_supported = _canonicalize_label(ground_truth)

    # Exact-match after canonicalization for single-label F1
    f1 = 1.0 if pred_canon == truth_canon else 0.0

    return {
        "f1_score": f1,
        "precision": f1,
        "recall": f1,
        "prediction_normalized": pred_canon.lower().strip(),
        "ground_truth_normalized": truth_canon.lower().strip(),
        "prediction_supported": pred_supported,
        "ground_truth_supported": truth_supported,
    }


def calculate_f1_stats(data_points):
    """Calculate both macro-F1 and average F1 (micro-F1) statistics.

    - Only supported classes are included in the class set
    - Unsupported predictions contribute FN to the ground-truth class but do not
      create or contribute FP to an unsupported predicted class
    """
    if not data_points:
        return {}

    # Calculate average F1 (micro-F1) - simple average across all predictions
    f1_scores = [point.get("f1_score", 0) for point in data_points]
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    # Initialize class buckets for only supported classes (lowercased for consistency)
    # Use discovered labels if available, otherwise fall back to dataset labels
    labels_to_use = SUPPORTED_LABELS if SUPPORTED_LABELS else FALLBACK_LABELS
    supported_lower = {label.lower(): label for label in labels_to_use}
    class_predictions = {
        lab.lower(): {"tp": 0, "fp": 0, "fn": 0} for lab in labels_to_use
    }

    for point in data_points:
        gt_class = point.get("ground_truth_normalized", "")
        pred_class = point.get("prediction_normalized", "")
        pred_supported = point.get("prediction_supported", False)

        # Ensure ground truth is one of the supported classes; if not, skip counting it
        if gt_class not in class_predictions:
            # Skip entirely as requested: do not include it in the ground truth labels
            continue

        if pred_class == gt_class:
            class_predictions[gt_class]["tp"] += 1
        else:
            # Count FN for the ground truth class
            class_predictions[gt_class]["fn"] += 1
            # Count FP only if predicted class is supported
            if pred_supported and pred_class in class_predictions:
                class_predictions[pred_class]["fp"] += 1

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

        # Use canonical casing in output keys
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


def parse_baseline_sleep_cot_json(input_file, output_file=None):
    """Parse baseline sleep COT JSON file and extract structured data."""
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}.clean.jsonl")

    print(f"Parsing {input_file}")
    print(f"Output will be saved to {output_file}")

    # First, discover the actual labels from the ground truth data
    global SUPPORTED_LABELS
    discovered_labels = discover_ground_truth_labels(input_file)
    SUPPORTED_LABELS = discovered_labels

    print(f"Discovered {len(discovered_labels)} labels from ground truth data:")
    for label in sorted(discovered_labels):
        print(f"  - {label}")

    extracted_data = extract_structured_data(input_file)

    if extracted_data:
        print(f"Extracted {len(extracted_data)} data points")

        # Calculate and display accuracy statistics
        accuracy_stats = calculate_accuracy_stats(extracted_data)
        print(f"\nAccuracy Statistics:")
        print(f"Total samples: {accuracy_stats['total_samples']}")
        print(f"Correct predictions: {accuracy_stats['correct_predictions']}")
        print(f"Incorrect predictions: {accuracy_stats['incorrect_predictions']}")
        print(f"Accuracy: {accuracy_stats['accuracy_percentage']:.2f}%")

        # Calculate and display F1 statistics
        f1_stats = calculate_f1_stats(extracted_data)
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

        with open(output_file, "w", encoding="utf-8") as f:
            for item in extracted_data:
                f.write(json.dumps(item, indent=2) + "\n")

        print(f"\nData saved to {output_file}")
        return extracted_data
    else:
        print("No data could be extracted from the file.")
        return []


def discover_ground_truth_labels(input_file):
    """Discover actual labels from ground truth data in the JSON file"""
    discovered_labels = set()

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

        # Navigate to detailed_results
        detailed_results = data.get("detailed_results", [])

        for result in detailed_results:
            target_answer = result.get("target_answer", "")
            ground_truth_raw = extract_answer(target_answer)
            gt_canon, _ = _canonicalize_label(ground_truth_raw)
            if gt_canon:
                discovered_labels.add(gt_canon)

    return list(discovered_labels)


def extract_structured_data(input_file):
    """Extract structured data from baseline JSON file"""
    data_points = []

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

        # Navigate to detailed_results
        detailed_results = data.get("detailed_results", [])

        for result in tqdm(detailed_results, desc="Processing results"):
            try:
                # Extract fields from the baseline JSON structure
                sample_idx = result.get("sample_idx", 0)
                generated_answer = result.get("generated_answer", "")
                target_answer = result.get("target_answer", "")

                # Extract answers from both fields
                model_prediction_raw = extract_answer(generated_answer)
                ground_truth_raw = extract_answer(target_answer)

                # Canonicalize labels and merge stage 4 -> stage 3
                pred_canon, pred_supported = _canonicalize_label(model_prediction_raw)
                gt_canon, gt_supported = _canonicalize_label(ground_truth_raw)

                # Calculate accuracy (exact match)
                accuracy = (pred_canon == gt_canon) and gt_supported

                # Calculate F1 score
                f1_result = calculate_f1_score(model_prediction_raw, ground_truth_raw)

                data_point = {
                    "sample_idx": sample_idx,
                    "generated": generated_answer,
                    "model_prediction": model_prediction_raw,
                    "ground_truth": ground_truth_raw,
                    "accuracy": accuracy,
                    "f1_score": f1_result["f1_score"],
                    "precision": f1_result["precision"],
                    "recall": f1_result["recall"],
                    "prediction_normalized": f1_result["prediction_normalized"],
                    "ground_truth_normalized": f1_result["ground_truth_normalized"],
                    "prediction_supported": f1_result["prediction_supported"],
                    "ground_truth_supported": f1_result["ground_truth_supported"],
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
        return text

    answer = text.split("Answer: ")[-1].strip()
    # Remove any end-of-text tokens (including <eos> and <|...|>)
    answer = re.sub(r"<\|.*?\|>|<eos>$", "", answer).strip()
    # Remove trailing periods and normalize
    answer = re.sub(r"\.$", "", answer).strip()
    return answer


if __name__ == "__main__":
    # Example usage with the baseline JSON file
    input_file = "evaluation_results_meta-llama-llama-3-2-3b_sleepedfcotqadataset.json"
    output_file = "out.jsonl"

    parse_baseline_sleep_cot_json(input_file, output_file)
