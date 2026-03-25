# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import re
import sys
from typing import Dict, Any, List, Tuple


from common_evaluator import CommonEvaluator
from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset


def extract_answer(text: str) -> str:
    """
    Extract the final answer from model text, following the parser rules:
    - If "Answer: " present, take substring after the last occurrence
    - Strip any special end tokens like <|...|> or <eos>
    - Trim trailing periods and whitespace
    """
    if text is None:
        return ""
    if "Answer: " not in text:
        return text.strip()
    answer = text.split("Answer: ")[-1].strip()
    answer = re.sub(r"<\|.*?\|>|<eos>$", "", answer).strip()
    answer = re.sub(r"\.$", "", answer).strip()
    return answer


def normalize_label(label: str) -> str:
    """Lowercase, strip, and remove trailing punctuation to match parser behavior."""
    if label is None:
        return ""
    return label.lower().strip().rstrip(".,!?;:")


def evaluate_ecg_metrics(
    ground_truth: str, prediction: str, sample: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Evaluate ECG-QA CoT predictions using per-template answers from CSV.
    Normalization matches the parser in evaluation/opentslm/ecg_qa_cot/parse_ecg_qa_cot_data.py.
    """
    # Extract answers
    pred_raw = extract_answer(prediction)
    gt_raw = extract_answer(ground_truth)

    # Normalize
    pred_norm = normalize_label(pred_raw)
    gt_norm = normalize_label(gt_raw)

    # Per-template supported answers (strict)
    if not isinstance(sample, dict):
        print(f"DEBUG: Sample type: {type(sample)}")
        print(f"DEBUG: Sample content: {sample}")
        raise ValueError(
            "Sample must be a dict containing 'template_id' for ECG-QA evaluation"
        )

    template_id = sample.get("template_id") or sample.get("cot_template_id")
    if template_id is None:
        print(f"DEBUG: Sample keys: {sample.keys()}")
        print(f"DEBUG: Sample content: {sample}")
        raise ValueError("Missing 'template_id' in sample for ECG-QA evaluation")

    possible_answers = ECGQACoTQADataset.get_possible_answers_for_template(
        int(template_id)
    )
    if not possible_answers:
        raise ValueError(f"No possible answers found for template_id={template_id}")

    possible_answers_lower = [a.lower().strip() for a in possible_answers]

    # Supported flags: restrict to template answers strictly
    pred_supported = pred_norm in possible_answers_lower
    gt_supported = gt_norm in possible_answers_lower

    # Exact match
    is_correct = int(pred_norm == gt_norm)

    # For single-label exact-match, precision=recall=F1=accuracy per-sample
    f1 = float(is_correct)

    return {
        "accuracy": is_correct,
        "f1_score": f1,
        "precision": f1,
        "recall": f1,
        "prediction_normalized": pred_norm,
        "ground_truth_normalized": gt_norm,
        "prediction_supported": pred_supported,
        "ground_truth_supported": gt_supported,
        "template_id": template_id,
        "possible_answers": possible_answers,
    }


# --- Parser-matching aggregation helpers ---


def _calculate_template_f1_stats(data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not data_points:
        return {}

    from collections import defaultdict

    template_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for point in data_points:
        template_id = point.get("template_id")
        if template_id is None:
            raise ValueError(f"Missing template_id in data point: {point}")
        template_groups[int(template_id)].append(point)

    template_stats: Dict[int, Dict[str, Any]] = {}
    total_samples = 0
    total_correct = 0
    total_f1_sum = 0.0

    for template_id, points in template_groups.items():
        if not points:
            continue

        possible_answers = points[0].get("possible_answers", [])
        if not possible_answers:
            raise ValueError(f"No possible answers found for template {template_id}")

        # Initialize per-class counts
        class_predictions: Dict[str, Dict[str, int]] = {}
        for answer in possible_answers:
            class_predictions[answer.lower()] = {"tp": 0, "fp": 0, "fn": 0}

        # Count TP/FP/FN
        for p in points:
            gt_class = p.get("ground_truth_normalized", "")
            pred_class = p.get("prediction_normalized", "")
            pred_supported = p.get("prediction_supported", False)

            if gt_class in class_predictions:
                if pred_class == gt_class:
                    class_predictions[gt_class]["tp"] += 1
                else:
                    class_predictions[gt_class]["fn"] += 1
                    if pred_supported and pred_class in class_predictions:
                        class_predictions[pred_class]["fp"] += 1

        # Per-class and macro-F1
        class_f1_scores: Dict[str, Dict[str, float]] = {}
        template_f1_sum = 0.0
        valid_classes = 0

        for class_name, counts in class_predictions.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
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
            template_f1_sum += f1
            valid_classes += 1

        macro_f1 = template_f1_sum / valid_classes if valid_classes > 0 else 0.0

        template_correct = sum(1 for p in points if p.get("accuracy", False))
        template_accuracy = template_correct / len(points) if points else 0.0
        template_avg_f1 = (
            sum(p.get("f1_score", 0.0) for p in points) / len(points) if points else 0.0
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
        total_f1_sum += template_avg_f1 * len(points)

    template_macro_f1s = [stats["macro_f1"] for stats in template_stats.values()]
    overall_macro_f1 = (
        sum(template_macro_f1s) / len(template_macro_f1s) if template_macro_f1s else 0.0
    )
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    overall_avg_f1 = total_f1_sum / total_samples if total_samples > 0 else 0.0

    return {
        "overall": {
            "total_samples": total_samples,
            "total_templates": len(template_stats),
            "accuracy": overall_accuracy,
            "average_f1": overall_avg_f1,
            "macro_f1": overall_macro_f1,
        },
        "per_template": template_stats,
    }


def _build_data_points_from_results(
    detailed_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    data_points: List[Dict[str, Any]] = []
    for r in detailed_results:
        m = r.get("metrics", {})
        template_id = m.get("template_id", r.get("template_id"))
        if template_id is None:
            raise ValueError("Missing template_id in detailed result")
        dp = {
            "template_id": int(template_id),
            "accuracy": m.get("accuracy", 0),
            "f1_score": m.get("f1_score", 0.0),
            "precision": m.get("precision", 0.0),
            "recall": m.get("recall", 0.0),
            "prediction_normalized": m.get("prediction_normalized", ""),
            "ground_truth_normalized": m.get("ground_truth_normalized", ""),
            "prediction_supported": m.get("prediction_supported", False),
            "ground_truth_supported": m.get("ground_truth_supported", False),
            "possible_answers": m.get("possible_answers", []),
        }
        if not dp["possible_answers"]:
            raise ValueError(
                f"No possible answers in metrics for template {template_id}"
            )
        data_points.append(dp)
    return data_points


def main():
    """Main function to run ECG-QA CoT evaluation with parser-matching F1 aggregation."""
    if len(sys.argv) != 2:
        print("Usage: python evaluate_ecg_qa.py <model_name>")
        print("Example: python evaluate_ecg_qa.py meta-llama/Llama-3.2-1B")
        sys.exit(1)

    model_name = sys.argv[1]

    evaluator = CommonEvaluator()
    # Run single evaluation to keep detailed results for F1 aggregation
    results = evaluator.evaluate_model_on_dataset(
        model_name=model_name,
        dataset_class=ECGQACoTQADataset,
        evaluation_function=evaluate_ecg_metrics,
        max_samples=490,
        max_new_tokens=400,
    )

    # Build data points and compute parser-matching F1 stats
    detailed_results = results.get("detailed_results", [])
    data_points = _build_data_points_from_results(detailed_results)
    f1_stats = _calculate_template_f1_stats(data_points)

    # Print parser-like summary
    overall = f1_stats.get("overall", {})
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY (Parser-matching)")
    print("=" * 80)
    print(f"Total templates: {overall.get('total_templates', 0)}")
    print(f"Average F1 Score: {overall.get('average_f1', 0):.4f}")
    print(f"Macro-F1 Score: {overall.get('macro_f1', 0):.4f}")

    per_template = f1_stats.get("per_template", {})
    if per_template:
        print(f"\nPer-Template Statistics:")
        for template_id, stats in sorted(per_template.items()):
            print(f"  Template {template_id}:")
            print(f"    Samples: {stats['num_samples']}")
            print(f"    Accuracy: {stats['accuracy']:.4f}")
            print(f"    Average F1: {stats['average_f1']:.4f}")
            print(f"    Macro-F1: {stats['macro_f1']:.4f}")

    return f1_stats


if __name__ == "__main__":
    main()
