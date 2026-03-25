#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Script to create doctor evaluation dataset with correct model predictions.
This script extracts ECG-QA templates with correct model outputs from llama3b_flamingo_predictions.jsonl
and creates organized folders with ECG plots, CSV data, and evaluation materials.
"""

import json
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from opentslm.time_series_datasets.ecg_qa.plot_example import get_ptbxl_ecg_path

# Configuration
MODEL_PREDICTIONS_FILE = "/Users/planger/Development/EmbedHealth/evaluation/opentslm/ecg_qa_cot/llama3b_flamingo_predictions.jsonl"
OUTPUT_DIR = "ecg_doctor_eval"
SAMPLES_PER_TEMPLATE = 2


def extract_answer_from_generated(generated_text: str) -> str:
    """Extract the final answer from generated text after 'Answer: '"""
    if "Answer: " not in generated_text:
        return generated_text.strip()

    answer = generated_text.split("Answer: ")[-1].strip()
    # Remove any end-of-text tokens and trailing punctuation
    answer = re.sub(r"<\|.*?\|>|<eos>$", "", answer).strip()
    answer = re.sub(r"\.$", "", answer).strip()
    return answer


def is_correct_prediction(generated_text: str, correct_answer: str) -> bool:
    """Check if the model prediction matches the correct answer"""
    predicted_answer = extract_answer_from_generated(generated_text)
    return predicted_answer.lower().strip() == correct_answer.lower().strip()


def load_model_predictions() -> Dict[int, List[Dict]]:
    """Load model predictions and group by template_id"""
    print(f"Loading model predictions from {MODEL_PREDICTIONS_FILE}")

    template_predictions = defaultdict(list)

    with open(MODEL_PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading predictions"), 1):
            try:
                data = json.loads(line.strip())

                template_id = data.get("template_id")
                if template_id is None:
                    continue

                # Check if prediction is correct
                generated_text = data.get("generated", "")
                correct_answer = data.get("correct_answer", "")

                if is_correct_prediction(generated_text, correct_answer):
                    template_predictions[template_id].append(
                        {
                            "template_id": template_id,
                            "ecg_id": data.get("ecg_id", [None])[0]
                            if data.get("ecg_id")
                            else None,
                            "generated": generated_text,
                            "correct_answer": correct_answer,
                            "pre_prompt": data.get("pre_prompt", ""),
                            "line_number": line_num,
                        }
                    )

            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    print(f"Found correct predictions for {len(template_predictions)} templates")
    return template_predictions


def extract_clinical_context(pre_prompt: str) -> str:
    """Extract clinical context from pre_prompt"""
    if "Clinical Context:" in pre_prompt:
        context_start = pre_prompt.find("Clinical Context:")
        context_end = pre_prompt.find("\n\n", context_start)
        if context_end == -1:
            context_end = pre_prompt.find("\n", context_start)
        if context_end != -1:
            return pre_prompt[context_start:context_end].strip()
    return "Clinical context not available"


def extract_question_from_prompt(pre_prompt: str) -> str:
    """Extract the question from pre_prompt - this is the authoritative question for each sample"""
    if "Question: " in pre_prompt:
        question_start = pre_prompt.find("Question: ")
        question_end = pre_prompt.find("\n\n", question_start)
        if question_end == -1:
            question_end = pre_prompt.find("\n", question_start)
        if question_end != -1:
            return pre_prompt[question_start:question_end].strip()
    return "Question not available"


def get_answer_options_for_template(template_id: int) -> List[str]:
    """Get answer options for a template"""
    try:
        return ECGQACoTQADataset.get_possible_answers_for_template(template_id)
    except Exception as e:
        print(f"Warning: Could not get answer options for template {template_id}: {e}")
        return []


def load_ecg_data(ecg_id: int) -> Tuple[np.ndarray, str]:
    """Load ECG data for a given ECG ID"""
    try:
        ecg_path = get_ptbxl_ecg_path(ecg_id)

        if not os.path.exists(ecg_path + ".dat"):
            raise FileNotFoundError(f"ECG file not found: {ecg_path}.dat")

        # Read ECG data using wfdb
        ecg_data, meta = wfdb.rdsamp(ecg_path)

        # Get sampling frequency
        sampling_freq = meta.get("fs", 500)  # Default to 500Hz if not specified

        return ecg_data, sampling_freq

    except Exception as e:
        raise RuntimeError(f"Failed to load ECG {ecg_id}: {e}")


def downsample_to_100hz(ecg_data: np.ndarray, original_freq: int) -> np.ndarray:
    """Downsample ECG data to 100Hz"""
    if original_freq == 100:
        return ecg_data

    # Calculate downsampling factor
    downsample_factor = original_freq // 100

    # Downsample by taking every nth sample
    downsampled_data = ecg_data[::downsample_factor]

    return downsampled_data


def save_ecg_as_csv(ecg_data: np.ndarray, output_dir: str, ecg_id: int):
    """Save ECG data as separate CSV files for each lead"""
    # Downsample to 100Hz if needed
    if ecg_data.shape[0] > 1000:  # Likely 500Hz data
        ecg_data = downsample_to_100hz(ecg_data, 500)

    lead_names = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]

    for lead_idx, lead_name in enumerate(lead_names):
        if lead_idx < ecg_data.shape[1]:  # Make sure we don't exceed available leads
            lead_data = ecg_data[:, lead_idx]

            # Create DataFrame with time and signal values
            time_points = np.arange(len(lead_data)) / 100.0  # 100Hz sampling
            df = pd.DataFrame({"time_seconds": time_points, "signal_mV": lead_data})

            # Save to CSV
            csv_filename = f"{output_dir}/lead_{lead_name}.csv"
            df.to_csv(csv_filename, index=False)


def create_ecg_plot(
    ecg_data: np.ndarray,
    template_id: int,
    ecg_id: int,
    question: str,
    answer_options: List[str],
    clinical_context: str,
    model_output: str,
    correct_answer: str,
    output_dir: str,
):
    """Create ECG plot with all information"""

    # Downsample to 100Hz if needed
    if ecg_data.shape[0] > 1000:  # Likely 500Hz data
        ecg_data = downsample_to_100hz(ecg_data, 500)

    # Create the plot with all 12 leads
    fig, axes = plt.subplots(12, 1, figsize=(14, 24))
    fig.suptitle(
        f"Template {template_id}: ECG Analysis\nECG ID: {ecg_id}",
        fontsize=16,
        fontweight="bold",
    )

    # Create time array for 100Hz sampling (10 seconds)
    time_points = np.arange(0, 10, 0.01)  # 100Hz for 10 seconds

    # Plot all 12 leads
    lead_names = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        if i < ecg_data.shape[1]:  # Make sure we don't exceed available leads
            # Plot the ECG signal for this lead
            ax.plot(time_points, ecg_data[:, i], linewidth=2, color="k", alpha=1.0)

            # Add grid lines (millimeter paper style)
            # Major grid lines (every 0.2s and 0.5mV)
            ax.vlines(
                np.arange(0, 10, 0.2), -2.5, 2.5, colors="r", alpha=0.3, linewidth=0.5
            )
            ax.hlines(
                np.arange(-2.5, 2.5, 0.5), 0, 10, colors="r", alpha=0.3, linewidth=0.5
            )

            # Minor grid lines (every 0.04s and 0.1mV)
            ax.vlines(
                np.arange(0, 10, 0.04), -2.5, 2.5, colors="r", alpha=0.1, linewidth=0.3
            )
            ax.hlines(
                np.arange(-2.5, 2.5, 0.1), 0, 10, colors="r", alpha=0.1, linewidth=0.3
            )

            ax.set_xticks(np.arange(0, 11, 1.0))
            ax.set_ylabel(f"Lead {lead_name} (mV)", fontweight="bold")
            ax.margins(0.0)
            ax.set_ylim(-2.5, 2.5)
            ax.set_title(f"Lead {lead_name}", fontweight="bold", pad=10)
        else:
            ax.set_title(f"Lead {lead_name} (not available)", fontweight="bold", pad=10)
            ax.text(
                0.5,
                0.5,
                "Lead not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    # Add information text box
    info_text = f"""Question: {question}

Answer Options: {" | ".join(answer_options[:5])}{"..." if len(answer_options) > 5 else ""}

Clinical Context: {clinical_context[:200]}{"..." if len(clinical_context) > 200 else ""}

Model Output: {model_output[:300]}{"..." if len(model_output) > 300 else ""}

Expected Answer: {correct_answer}"""

    fig.text(
        0.02,
        0.02,
        info_text,
        fontsize=9,
        transform=fig.transFigure,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    # Save plot
    plot_filename = f"{output_dir}/ecg_plot.png"
    fig.savefig(plot_filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return plot_filename


def create_evaluation_text_file(
    output_dir: str,
    template_id: int,
    ecg_id: int,
    question: str,
    answer_options: List[str],
    clinical_context: str,
    model_output: str,
    correct_answer: str,
):
    """Create a text file with all evaluation information"""

    txt_filename = f"{output_dir}/evaluation_info.txt"
    with open(txt_filename, "w") as f:
        f.write(f"ECG-QA Doctor Evaluation\n")
        f.write(f"=" * 50 + "\n\n")

        f.write(f"Template ID: {template_id}\n")
        f.write(f"ECG ID: {ecg_id}\n\n")

        f.write(f"Question:\n{question}\n\n")

        f.write(f"Answer Options:\n")
        for i, option in enumerate(answer_options, 1):
            f.write(f"{i}. {option}\n")
        f.write(f"\n")

        f.write(f"Clinical Context:\n{clinical_context}\n\n")

        f.write(f"Model Output (Llama3B-Flamingo):\n{model_output}\n\n")

        f.write(f"Expected Answer: {correct_answer}\n")


def create_doctor_evaluation_dataset():
    """Main function to create the doctor evaluation dataset"""

    print("Creating doctor evaluation dataset...")
    print(
        "Note: Each template_id can have multiple different questions - using the specific question from each sample"
    )

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model predictions
    template_predictions = load_model_predictions()

    if not template_predictions:
        print("No correct predictions found!")
        return

    # Process each template
    processed_templates = 0

    for template_id in sorted(template_predictions.keys()):
        predictions = template_predictions[template_id]

        if len(predictions) < SAMPLES_PER_TEMPLATE:
            print(
                f"Template {template_id}: Only {len(predictions)} correct predictions, skipping"
            )
            continue

        print(
            f"\nProcessing template {template_id} with {len(predictions)} correct predictions"
        )

        # Get answer options for this template
        answer_options = get_answer_options_for_template(template_id)
        if not answer_options:
            print(f"Template {template_id}: No answer options found, skipping")
            continue

        # Process up to SAMPLES_PER_TEMPLATE samples
        samples_to_process = predictions[:SAMPLES_PER_TEMPLATE]

        for sample_idx, prediction in enumerate(samples_to_process, 1):
            try:
                ecg_id = prediction["ecg_id"]
                if ecg_id is None:
                    print(
                        f"Template {template_id}, Sample {sample_idx}: No ECG ID, skipping"
                    )
                    continue

                print(f"  Processing sample {sample_idx}: ECG {ecg_id}")

                # Create sample directory
                sample_dir = (
                    f"{OUTPUT_DIR}/template_{template_id:02d}/sample{sample_idx}"
                )
                os.makedirs(sample_dir, exist_ok=True)

                # Extract information
                clinical_context = extract_clinical_context(prediction["pre_prompt"])
                question = extract_question_from_prompt(
                    prediction["pre_prompt"]
                )  # Use the specific question from this sample
                model_output = prediction["generated"]
                correct_answer = prediction["correct_answer"]

                # Load ECG data
                try:
                    ecg_data, sampling_freq = load_ecg_data(ecg_id)
                    print(f"    Loaded ECG data: {ecg_data.shape}, {sampling_freq}Hz")
                except Exception as e:
                    print(f"    Error loading ECG {ecg_id}: {e}")
                    continue

                # Save ECG as CSV files
                try:
                    save_ecg_as_csv(ecg_data, sample_dir, ecg_id)
                    print(f"    Saved ECG CSV files")
                except Exception as e:
                    print(f"    Error saving ECG CSV: {e}")

                # Create ECG plot
                try:
                    plot_filename = create_ecg_plot(
                        ecg_data,
                        template_id,
                        ecg_id,
                        question,
                        answer_options,
                        clinical_context,
                        model_output,
                        correct_answer,
                        sample_dir,
                    )
                    print(f"    Created ECG plot: {plot_filename}")
                except Exception as e:
                    print(f"    Error creating ECG plot: {e}")

                # Create evaluation text file
                try:
                    create_evaluation_text_file(
                        sample_dir,
                        template_id,
                        ecg_id,
                        question,
                        answer_options,
                        clinical_context,
                        model_output,
                        correct_answer,
                    )
                    print(f"    Created evaluation text file")
                except Exception as e:
                    print(f"    Error creating text file: {e}")

            except Exception as e:
                print(f"    Error processing sample {sample_idx}: {e}")
                continue

        processed_templates += 1
        print(f"Completed template {template_id}")

    print(f"\nDoctor evaluation dataset creation completed!")
    print(f"Processed {processed_templates} templates")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create summary file
    create_summary_file(template_predictions)


def create_summary_file(template_predictions: Dict[int, List[Dict]]):
    """Create a summary file with statistics"""
    summary_file = f"{OUTPUT_DIR}/dataset_summary.txt"

    with open(summary_file, "w") as f:
        f.write("ECG-QA Doctor Evaluation Dataset Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(
            f"Total templates with correct predictions: {len(template_predictions)}\n"
        )
        f.write(f"Samples per template: {SAMPLES_PER_TEMPLATE}\n")
        f.write(
            f"Total samples created: {len(template_predictions) * SAMPLES_PER_TEMPLATE}\n\n"
        )

        f.write("Template Statistics:\n")
        f.write("-" * 30 + "\n")

        for template_id in sorted(template_predictions.keys()):
            predictions = template_predictions[template_id]
            f.write(
                f"Template {template_id:2d}: {len(predictions):3d} correct predictions\n"
            )

        f.write(f"\nDataset Structure:\n")
        f.write(f"ecg_doctor_eval/\n")
        f.write(f"├── template_01/\n")
        f.write(f"│   ├── sample1/\n")
        f.write(f"│   │   ├── ecg_plot.png\n")
        f.write(f"│   │   ├── evaluation_info.txt\n")
        f.write(f"│   │   ├── lead_I.csv\n")
        f.write(f"│   │   ├── lead_II.csv\n")
        f.write(f"│   │   └── ... (all 12 leads)\n")
        f.write(f"│   └── sample2/\n")
        f.write(f"│       └── ... (same structure)\n")
        f.write(f"├── template_02/\n")
        f.write(f"│   └── ...\n")
        f.write(f"└── dataset_summary.txt\n")

        f.write(f"\nNotes:\n")
        f.write(
            f"- All predictions are CORRECT (model answer matches expected answer)\n"
        )
        f.write(f"- ECG data is downsampled to 100Hz for consistency\n")
        f.write(
            f"- Each sample includes clinical context, question, answer options, and model reasoning\n"
        )
        f.write(f"- CSV files contain time series data for each ECG lead\n")

    print(f"Summary file created: {summary_file}")


if __name__ == "__main__":
    try:
        create_doctor_evaluation_dataset()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
