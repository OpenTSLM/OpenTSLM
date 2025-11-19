#!/usr/bin/env python3
"""
Evaluate ECG-QA inference results from a CSV file.
Reuses evaluation logic from evaluate_ecg_qa.py.
"""
import argparse
import pandas as pd

from evaluate_ecg_qa import (
    evaluate_ecg_metrics,
    _calculate_template_f1_stats,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ECG-QA inference results from CSV")
    parser.add_argument("csv_path", type=str, help="Path to inference results CSV")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-sample details")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} samples from {args.csv_path}\n")

    # Evaluate each sample using imported function
    data_points = []
    for _, row in df.iterrows():
        sample = {"template_id": row["template_id"]}
        metrics = evaluate_ecg_metrics(
            row["target_answer"],
            row["generated_answer"],
            sample
        )
        data_points.append(metrics)

    # Aggregate using imported function
    f1_stats = _calculate_template_f1_stats(data_points)

    # Print results
    overall = f1_stats.get("overall", {})
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total samples evaluated: {overall.get('total_samples', 0)}")
    print(f"Accuracy: {overall.get('accuracy', 0):.4f}")
    print(f"Average F1 Score: {overall.get('average_f1', 0):.4f}")
    print(f"Macro-F1 Score: {overall.get('macro_f1', 0):.4f}")

    # Report skipped templates
    skipped_templates = overall.get('skipped_templates', 0)
    skipped_samples = overall.get('skipped_samples', 0)
    if skipped_templates > 0:
        print(f"\nWarning: Skipped {skipped_templates} templates ({skipped_samples} samples) due to missing answers")
        skipped_details = f1_stats.get("skipped_template_details", [])
        for template_id, count in skipped_details:
            print(f"  Template {template_id}: {count} samples skipped")

    # Per-template stats
    per_template = f1_stats.get("per_template", {})
    if per_template:
        print(f"\nPer-Template Statistics:")
        for template_id, stats in sorted(per_template.items()):
            print(f"  Template {template_id}:")
            print(f"    Samples: {stats['num_samples']}")
            print(f"    Accuracy: {stats['accuracy']:.4f}")
            print(f"    Macro-F1: {stats['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
