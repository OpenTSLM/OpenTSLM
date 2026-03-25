# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import json
import os


def first_three(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip()[:3]


def calculate_f1_score(prediction: str, ground_truth: str):
    pred_normalized = first_three(prediction).lower()
    truth_normalized = first_three(ground_truth).lower()
    f1 = 1.0 if pred_normalized == truth_normalized else 0.0
    return {
        "f1_score": f1,
        "precision": f1,
        "recall": f1,
        "prediction_normalized": pred_normalized,
        "ground_truth_normalized": truth_normalized,
    }


def calculate_f1_stats(data_points, allowed_labels=None):
    if not data_points:
        return {}
    f1_scores = [p.get("f1_score", 0) for p in data_points]
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    class_predictions = {}
    if allowed_labels:
        for label in allowed_labels:
            class_predictions[label] = {"tp": 0, "fp": 0, "fn": 0}
    for p in data_points:
        gt = p.get("ground_truth_normalized", "")
        pr = p.get("prediction_normalized", "")
        if gt not in class_predictions:
            class_predictions[gt] = {"tp": 0, "fp": 0, "fn": 0}
        if pr == gt:
            class_predictions[gt]["tp"] += 1
        else:
            class_predictions[gt]["fn"] += 1
            if (allowed_labels is None) or (pr in (allowed_labels or set())):
                if pr in class_predictions:
                    class_predictions[pr]["fp"] += 1
                else:
                    class_predictions[pr] = {"tp": 0, "fp": 1, "fn": 0}

    class_f1_scores = {}
    total_f1 = 0
    valid_classes = 0
    for cls, c in class_predictions.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        class_f1_scores[cls] = {
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


def parse_baseline_json(input_path: str):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detailed = data.get("detailed_results", [])
    total = 0
    correct = 0
    data_points = []

    labels = ["(a)", "(b)", "(c)"]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    confusion = [[0, 0, 0] for _ in range(3)]
    support = {l: 0 for l in labels}

    for i, item in enumerate(detailed):
        gold_raw = item.get("target_answer", "")
        pred_raw = item.get("generated_answer", "")

        gold = first_three(gold_raw)
        pred = first_three(pred_raw)

        total += 1
        is_correct = gold == pred
        if is_correct:
            correct += 1
        else:
            print(f"Line {i} - Pred: {pred_raw} -> {pred}, Gold: {gold_raw} -> {gold}")

        f1_result = calculate_f1_score(pred, gold)
        data_points.append(
            {
                "accuracy": is_correct,
                "f1_score": f1_result["f1_score"],
                "precision": f1_result["precision"],
                "recall": f1_result["recall"],
                "prediction_normalized": f1_result["prediction_normalized"],
                "ground_truth_normalized": f1_result["ground_truth_normalized"],
            }
        )

        if gold in label_to_idx and pred in label_to_idx:
            gi = label_to_idx[gold]
            pi = label_to_idx[pred]
            confusion[gi][pi] += 1
            support[gold] += 1

    if total == 0:
        print("No valid entries found.")
        return

    accuracy = correct / total
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})")

    allowed_labels = {p.get("ground_truth_normalized", "") for p in data_points}
    f1_stats = calculate_f1_stats(data_points, allowed_labels=allowed_labels)
    print(f"\nF1 Score Statistics:")
    print(f"Average F1 Score: {f1_stats['average_f1']:.4f}")
    print(f"Macro-F1 Score: {f1_stats['macro_f1']:.4f}")
    print(f"Total Classes: {f1_stats['total_classes']}")

    if f1_stats["class_f1_scores"]:
        print(f"\nPer-Class F1 Scores:")
        for cls, scores in f1_stats["class_f1_scores"].items():
            print(
                f"  {cls}: F1={scores['f1']:.4f}, P={scores['precision']:.4f}, R={scores['recall']:.4f}"
            )

    print("\nClass support (gold counts):")
    for l in labels:
        print(f"  {l}: {support.get(l, 0)}")

    print("\nConfusion matrix (rows=gold, cols=pred):")
    header = "       " + "  ".join(labels)
    print(header)
    for i, l in enumerate(labels):
        row = "  ".join(str(x) for x in confusion[i])
        print(f"  {l}  {row}")


if __name__ == "__main__":
    # Default path: update if needed
    input_file = "evaluation_results_meta-llama-llama-3-2-3b_tsqadataset.json"
    parse_baseline_json(input_file)
