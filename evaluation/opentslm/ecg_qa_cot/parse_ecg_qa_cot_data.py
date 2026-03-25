#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Parser for converting ECG-QA CoT JSONL files to clean format with per-template metrics."""

import json
import re
import sys
import os
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

# Import dataset via package namespace
from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset


def calculate_f1_score(prediction, ground_truth, possible_answers):
    """Calculate F1 score for single-label classification with template-specific answers.

    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        possible_answers: List of valid answers for this template

    Returns:
        Dict with F1 metrics and metadata
    """
    if prediction is None:
        raise ValueError("Prediction cannot be None")
    if ground_truth is None:
        raise ValueError("Ground truth cannot be None")
    if not possible_answers:
        raise ValueError("Possible answers list cannot be empty")

    # Normalize predictions and ground truth
    pred_normalized = prediction.lower().strip().rstrip(".,!?;:")
    truth_normalized = ground_truth.lower().strip().rstrip(".,!?;:")

    # Check if prediction is in supported answers
    possible_answers_lower = [ans.lower().strip() for ans in possible_answers]
    pred_supported = pred_normalized in possible_answers_lower
    truth_supported = truth_normalized in possible_answers_lower

    # Calculate F1 (exact match after normalization)
    f1 = 1.0 if pred_normalized == truth_normalized else 0.0

    return {
        "f1_score": f1,
        "precision": f1,
        "recall": f1,
        "prediction_normalized": pred_normalized,
        "ground_truth_normalized": truth_normalized,
        "prediction_supported": pred_supported,
        "ground_truth_supported": truth_supported,
        "possible_answers": possible_answers,
    }


def calculate_template_f1_stats(data_points):
    """Calculate F1 statistics per template.

    Args:
        data_points: List of data points with template_id and metrics

    Returns:
        Dict with per-template and overall statistics
    """
    if not data_points:
        return {}

    # Group by template_id
    template_groups = defaultdict(list)
    for point in data_points:
        template_id = point.get("template_id")
        if template_id is None:
            raise ValueError(f"Missing template_id in data point: {point}")
        template_groups[template_id].append(point)

    # Calculate per-template metrics
    template_stats = {}
    total_samples = 0
    total_correct = 0
    total_f1_sum = 0
    total_macro_f1_weighted_sum = 0

    for template_id, points in template_groups.items():
        if not points:
            continue

        # Get possible answers for this template - required for evaluation
        possible_answers = points[0].get("possible_answers", [])
        if not possible_answers:
            raise ValueError(f"No possible answers found for template {template_id}")

        # Calculate per-template F1 stats
        class_predictions = {}
        for answer in possible_answers:
            class_predictions[answer.lower()] = {"tp": 0, "fp": 0, "fn": 0}

        # Count TP, FP, FN for each class in this template
        for point in points:
            gt_class = point.get("ground_truth_normalized", "")
            pred_class = point.get("prediction_normalized", "")
            pred_supported = point.get("prediction_supported", False)

            # Only count if ground truth is in supported answers
            if gt_class in class_predictions:
                if pred_class == gt_class:
                    class_predictions[gt_class]["tp"] += 1
                else:
                    # Count FN for ground truth class
                    class_predictions[gt_class]["fn"] += 1
                    # Count FP only if prediction is supported
                    if pred_supported and pred_class in class_predictions:
                        class_predictions[pred_class]["fp"] += 1

        # Calculate per-class F1 for this template
        class_f1_scores = {}
        template_f1_sum = 0
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

            template_f1_sum += f1
            valid_classes += 1

        # Calculate macro-F1 for this template
        macro_f1 = template_f1_sum / valid_classes if valid_classes > 0 else 0

        # Calculate accuracy for this template
        template_correct = sum(1 for point in points if point.get("accuracy", False))
        template_accuracy = template_correct / len(points) if points else 0

        # Calculate average F1 for this template
        template_avg_f1 = (
            sum(point.get("f1_score", 0) for point in points) / len(points)
            if points
            else 0
        )

        template_stats[template_id] = {
            "num_samples": len(points),
            "accuracy": template_accuracy,
            "average_f1": template_avg_f1,
            "macro_f1": macro_f1,
            "class_f1_scores": class_f1_scores,
            "num_classes": valid_classes,
            "correct_predictions": template_correct,
        }

        total_samples += len(points)
        total_correct += template_correct
        total_f1_sum += template_avg_f1 * len(points)  # Weighted by number of samples
        total_macro_f1_weighted_sum += macro_f1 * len(points)

    # Calculate overall statistics
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    overall_avg_f1 = total_f1_sum / total_samples if total_samples > 0 else 0

    # Calculate macro-F1 across templates
    template_macro_f1s = [stats["macro_f1"] for stats in template_stats.values()]
    overall_macro_f1 = (
        sum(template_macro_f1s) / len(template_macro_f1s) if template_macro_f1s else 0
    )
    overall_macro_f1_weighted = (
        total_macro_f1_weighted_sum / total_samples if total_samples > 0 else 0
    )

    # Calculate unweighted average of per-template accuracies
    template_accuracies = [stats["accuracy"] for stats in template_stats.values()]
    overall_template_accuracy_avg = (
        sum(template_accuracies) / len(template_accuracies)
        if template_accuracies
        else 0
    )

    return {
        "overall": {
            "total_samples": total_samples,
            "total_templates": len(template_stats),
            "accuracy": overall_accuracy,
            "template_accuracy_avg": overall_template_accuracy_avg,
            "average_f1": overall_avg_f1,
            "macro_f1": overall_macro_f1,
            "macro_f1_weighted": overall_macro_f1_weighted,
        },
        "per_template": template_stats,
    }


def calculate_accuracy_stats(data_points):
    """Calculate overall accuracy statistics from data points"""
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


def parse_ecg_qa_cot_jsonl(input_file, output_file=None):
    """Parse ECG-QA CoT JSONL file and extract JSON objects with per-template metrics."""
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}.clean.jsonl")

    print(f"Parsing {input_file}")
    print(f"Output will be saved to {output_file}")

    extracted_data = extract_structured_data(input_file)

    if extracted_data:
        print(f"Extracted {len(extracted_data)} data points")

        # Calculate and display overall accuracy statistics
        accuracy_stats = calculate_accuracy_stats(extracted_data)
        print(f"\nOverall Accuracy Statistics:")
        print(f"Total samples: {accuracy_stats['total_samples']}")
        print(f"Correct predictions: {accuracy_stats['correct_predictions']}")
        print(f"Incorrect predictions: {accuracy_stats['incorrect_predictions']}")
        print(f"Accuracy: {accuracy_stats['accuracy_percentage']:.2f}%")

        # Calculate and display per-template F1 statistics
        f1_stats = calculate_template_f1_stats(extracted_data)
        print(f"\nOverall F1 Statistics:")
        overall = f1_stats.get("overall", {})
        print(f"Total templates: {overall.get('total_templates', 0)}")
        print(f"Average F1 Score (sample-weighted): {overall.get('average_f1', 0):.4f}")
        print(
            f"Macro-F1 Score (unweighted over templates): {overall.get('macro_f1', 0):.4f}"
        )
        print(
            f"Macro-F1 Score (sample-weighted over templates): {overall.get('macro_f1_weighted', 0):.4f}"
        )
        print(
            f"Template Accuracy Avg (unweighted): {overall.get('template_accuracy_avg', 0):.4f}"
        )

        # Final single-value summary (aggregated across all templates)
        print("\nFinal Results (aggregated across all templates):")
        print(f"Final Accuracy (micro over samples): {overall.get('accuracy', 0):.4f}")
        print(f"Final F1 (micro over samples): {overall.get('accuracy', 0):.4f}")
        print(
            f"Final Macro-F1 (weighted by template size): {overall.get('macro_f1_weighted', 0):.4f}"
        )

        # Display per-template statistics
        per_template = f1_stats.get("per_template", {})
        if per_template:
            print(f"\nPer-Template Statistics:")
            for template_id, stats in sorted(per_template.items()):
                print(f"  Template {template_id}:")
                print(f"    Samples: {stats['num_samples']}")
                print(f"    Accuracy: {stats['accuracy']:.4f}")
                print(f"    Average F1: {stats['average_f1']:.4f}")
                print(f"    Macro-F1: {stats['macro_f1']:.4f}")

                # Show per-class F1 scores for this template
                if stats["class_f1_scores"]:
                    print(f"    Per-class F1:")
                    for class_name, scores in stats["class_f1_scores"].items():
                        if (
                            scores["tp"] + scores["fp"] + scores["fn"] > 0
                        ):  # Only show classes with samples
                            print(
                                f"      {class_name}: F1={scores['f1']:.4f}, P={scores['precision']:.4f}, R={scores['recall']:.4f}"
                            )

        with open(output_file, "w", encoding="utf-8") as f:
            for item in extracted_data:
                f.write(json.dumps(item, indent=2) + "\n")

        print(f"\nData saved to {output_file}")

        # Print concise overall stats at the end as a final summary
        print("\n==== Final Summary (end of run) ====")
        print(f"Samples: {accuracy_stats['total_samples']}")
        print(f"Accuracy (micro): {overall.get('accuracy', 0):.4f}")
        print(f"F1 (micro): {overall.get('accuracy', 0):.4f}")
        print(f"Macro-F1 (unweighted): {overall.get('macro_f1', 0):.4f}")
        print(f"Macro-F1 (weighted): {overall.get('macro_f1_weighted', 0):.4f}")
        print(
            f"Template Accuracy Avg (unweighted): {overall.get('template_accuracy_avg', 0):.4f}"
        )
        return extracted_data
    else:
        print("No data could be extracted from the file.")
        return []


def extract_structured_data(input_file):
    """Extract structured data from JSONL file"""
    data_points = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in tqdm(enumerate(f, 1), desc="Processing JSONL"):
            try:
                # Parse JSON line
                data = json.loads(line.strip())

                # Extract generated and gold fields - these are required
                generated_text = data.get("generated_answer")
                gold_text = data.get("target_answer")

                if generated_text is None:
                    raise ValueError(f"Missing 'generated' field in line {line_num}")
                if gold_text is None:
                    raise ValueError(f"Missing 'gold' field in line {line_num}")

                # Extract template_id and ecg_id - these are required for ECG-QA evaluation
                template_id = data.get("template_id")
                ecg_id = data.get("ecg_id", "None")

                if template_id is None:
                    raise ValueError(f"Missing template_id in line {line_num}")
                if ecg_id is None:
                    raise ValueError(f"Missing ecg_id in line {line_num}")

                # Extract answers from both fields
                model_prediction_raw = extract_answer(generated_text)
                ground_truth_raw = extract_answer(gold_text)

                # Get possible answers for this template - required for evaluation
                try:
                    possible_answers = (
                        ECGQACoTQADataset.get_possible_answers_for_template(template_id)
                    )
                except Exception as e:
                    raise ValueError(
                        f"Could not get possible answers for template {template_id}: {e}"
                    )

                if not possible_answers:
                    raise ValueError(
                        f"No possible answers found for template {template_id}"
                    )

                # Calculate F1 score with template-specific answers
                f1_result = calculate_f1_score(
                    model_prediction_raw, ground_truth_raw, possible_answers
                )

                # Calculate accuracy (exact match)
                accuracy = (
                    f1_result["prediction_normalized"]
                    == f1_result["ground_truth_normalized"]
                ) and f1_result["ground_truth_supported"]

                data_point = {
                    "generated": generated_text,
                    "model_prediction": model_prediction_raw,
                    "ground_truth": ground_truth_raw,
                    "template_id": template_id,
                    "ecg_id": ecg_id,
                    "accuracy": accuracy,
                    "f1_score": f1_result["f1_score"],
                    "precision": f1_result["precision"],
                    "recall": f1_result["recall"],
                    "prediction_normalized": f1_result["prediction_normalized"],
                    "ground_truth_normalized": f1_result["ground_truth_normalized"],
                    "prediction_supported": f1_result["prediction_supported"],
                    "ground_truth_supported": f1_result["ground_truth_supported"],
                    "possible_answers": possible_answers,
                    "line_number": line_num,
                }
                data_points.append(data_point)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
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
    current_dir = Path(__file__).parent
    input_file = (
        current_dir / "evaluation_results_openai-gpt-4o_ecgqacotqadataset.jsonl"
    )
    clean_output = (
        current_dir / "evaluation_results_openai-gpt-4o_ecgqacotqadataset.clean.jsonl"
    )

    parse_ecg_qa_cot_jsonl(input_file, clean_output)
