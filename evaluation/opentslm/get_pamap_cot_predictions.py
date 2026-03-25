#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Script to evaluate PAMAP2CoTQADataset with a trained OpenTSLMFlamingo model.
Stores time series data, ground truth labels, and rationale to CSV for later plotting.

Usage:
    python plot_pamap_cot_predictions.py

Requirements:
    - A trained OpenTSLMFlamingo model saved as a .pt file
    - The PAMAP2CoTQADataset should be available
    - Required dependencies: torch, pandas, numpy

Output:
    - CSV file with time series data, ground truth labels, and rationale
"""

import torch
import pandas as pd
import random
from typing import List, Dict, Any
import json


from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


def setup_device():
    """Setup the device for model inference."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device


def load_model(model_path: str, device: str, llm_id: str = "meta-llama/Llama-3.2-1B"):
    """Load the trained OpenTSLMFlamingo model."""
    print(f"Loading model from {model_path}...")

    model = OpenTSLMFlamingo(
        device=device,
        llm_id=llm_id,
        cross_attn_every_n_layers=1,
    )

    model.load_from_file(model_path)
    model.eval()
    print("✅ Model loaded successfully")
    return model


def load_dataset(split: str = "test"):
    """Load the PAMAP2CoTQADataset."""
    print(f"Loading PAMAP2CoTQADataset ({split} split)...")

    dataset = PAMAP2CoTQADataset(split=split, EOS_TOKEN="", min_series_length=150)

    print(f"✅ Dataset loaded with {len(dataset)} samples")
    return dataset


def run_inference_and_collect_data(
    model: OpenTSLMFlamingo,
    dataset: PAMAP2CoTQADataset,
    num_samples: int = 10,
    max_new_tokens: int = 300,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run inference on random samples and collect time series data, labels, and rationale."""
    print(f"Collecting data from {num_samples} random samples...")

    # Set random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Select random indices
    dataset_size = len(dataset)
    selected_indices = random.sample(
        range(dataset_size), min(num_samples, dataset_size)
    )

    results = []

    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            print(f"Processing sample {i + 1}/{len(selected_indices)} (index {idx})...")

            # Get the sample
            row = dataset[idx]

            # Extract raw time series data
            x_axis = row.get("x_axis", [])
            y_axis = row.get("y_axis", [])
            z_axis = row.get("z_axis", [])

            # Get ground truth label and rationale
            ground_truth_label = row["label"]
            rationale = row["answer"]

            # Run inference to get prediction
            try:
                # Build the prompt for inference
                pre_prompt = TextPrompt(row["pre_prompt"])
                post_prompt = TextPrompt(row["post_prompt"])

                # Create time series prompts using the data from the dataset
                ts_prompts = []
                for ts_text, ts_data in zip(
                    row["time_series_text"], row["time_series"]
                ):
                    ts_prompts.append(TextTimeSeriesPrompt(ts_text, ts_data))

                # Create full prompt
                prompt = FullPrompt(pre_prompt, ts_prompts, post_prompt)

                # Run inference
                prediction = model.eval_prompt(prompt, max_new_tokens=max_new_tokens)
                predicted_label = extract_activity_label(prediction)

                result = {
                    "sample_index": idx,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "z_axis": z_axis,
                    "ground_truth_label": ground_truth_label,
                    "predicted_label": predicted_label,
                    "rationale": rationale,
                    "full_prediction": prediction,
                    "series_length": len(x_axis),
                }

                results.append(result)
                print(f"  Ground truth: {ground_truth_label}")
                print(f"  Prediction: {predicted_label}")

            except Exception as e:
                print(f"  ❌ Error processing sample {idx}: {e}")
                continue

    print(f"✅ Successfully collected data from {len(results)} samples")
    return results


def extract_activity_label(prediction: str) -> str:
    """Extract the activity label from the model prediction."""
    # Look for "Answer: " pattern
    if "Answer:" in prediction:
        # Extract everything after "Answer: "
        answer_part = prediction.split("Answer:")[-1].strip()
        # Take the first word as the activity label
        label = answer_part.split()[0].strip().lower()
        return label
    else:
        # If no "Answer:" pattern, try to extract the last word as the activity
        words = prediction.strip().split()
        if words:
            return words[-1].strip().lower()
        else:
            return "unknown"


def save_results_to_csv(results: List[Dict[str, Any]], output_path: str):
    """Save the results to a CSV file."""
    print(f"Saving results to {output_path}...")

    # Prepare data for CSV - convert lists to JSON strings for better CSV handling
    csv_data = []
    for result in results:
        csv_row = {
            "sample_index": result["sample_index"],
            "x_axis": json.dumps(result["x_axis"]),
            "y_axis": json.dumps(result["y_axis"]),
            "z_axis": json.dumps(result["z_axis"]),
            "ground_truth_label": result["ground_truth_label"],
            "predicted_label": result["predicted_label"],
            "rationale": result["rationale"],
            "full_prediction": result["full_prediction"],
            "series_length": result["series_length"],
        }
        csv_data.append(csv_row)

    # Convert results to DataFrame
    df = pd.DataFrame(csv_data)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")

    # Print summary
    print(f"\n📊 Summary:")
    print(f"Total samples: {len(results)}")
    correct = sum(1 for r in results if r["ground_truth_label"] == r["predicted_label"])
    accuracy = correct / len(results) if results else 0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(results)})")


def main():
    """Main function to run the evaluation."""
    print("🚀 Starting PAMAP2CoTQADataset data collection...")
    print("=" * 60)

    # Configuration - adjust these parameters as needed
    config = {
        "model_path": "best_model.pt",  # Path to your trained model
        "output_path": "pamap_cot_data.csv",  # Output CSV file
        "num_samples": 10,  # Number of random samples to evaluate
        "llm_id": "meta-llama/Llama-3.2-1B",  # LLM ID used for training
        "dataset_split": "test",  # Dataset split to use: "train", "validation", or "test"
        "max_new_tokens": 300,  # Maximum tokens to generate
        "random_seed": 42,  # Random seed for reproducibility
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Setup
    device = setup_device()

    # Load model
    model = load_model(config["model_path"], device, config["llm_id"])

    # Load dataset
    dataset = load_dataset(split=config["dataset_split"])

    # Run inference and collect data
    results = run_inference_and_collect_data(
        model,
        dataset,
        config["num_samples"],
        config["max_new_tokens"],
        config["random_seed"],
    )

    # Save results
    save_results_to_csv(results, config["output_path"])

    print("🎉 Data collection completed successfully!")


if __name__ == "__main__":
    main()
