#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
ECG-QA Example Plot Generator
Creates a proper ECG visualization on millimeter paper grid and shows the corresponding prompt.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import wfdb

from opentslm.time_series_datasets.ecg_qa.ECGQADataset import ECGQADataset

# ECG plotting configuration
lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
hz = 500
start_time = 0
time = 10
start_length = int(start_time * hz)
sample_length = int(time * hz)
end_time = start_time + time
t = np.arange(start_time, end_time, 1 / hz)


def draw_ecg(ecg, lead=1, ax=None):
    """Draw a single ECG lead with millimeter paper grid."""
    if ax is None:
        ax = plt.gca()

    # Plot ECG signal
    ax.plot(
        t,
        ecg[lead][start_length : start_length + sample_length],
        linewidth=2,
        color="k",
        alpha=1.0,
        label=lead_names[lead],
    )

    # Calculate appropriate y-limits
    minimum = min(ecg[lead])
    maximum = max(ecg[lead])
    ylims_candidates = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]

    ylims = (
        max([x for x in ylims_candidates if x <= minimum]),
        min([x for x in ylims_candidates if x >= maximum]),
    )

    # Draw millimeter paper grid
    # Major grid lines (every 0.2s and 0.5mV)
    ax.vlines(
        np.arange(start_time, end_time, 0.2),
        ylims[0],
        ylims[1],
        colors="r",
        alpha=1.0,
        linewidth=0.8,
    )
    ax.hlines(
        np.arange(ylims[0], ylims[1], 0.5),
        start_time,
        end_time,
        colors="r",
        alpha=1.0,
        linewidth=0.8,
    )

    # Minor grid lines (every 0.04s and 0.1mV)
    ax.vlines(
        np.arange(start_time, end_time, 0.04),
        ylims[0],
        ylims[1],
        colors="r",
        alpha=0.3,
        linewidth=0.4,
    )
    ax.hlines(
        np.arange(ylims[0], ylims[1], 0.1),
        start_time,
        end_time,
        colors="r",
        alpha=0.3,
        linewidth=0.4,
    )

    ax.set_xticks(np.arange(start_time, end_time + 1, 1.0))
    ax.set_ylabel(f"Lead {lead_names[lead]} (mV)", fontweight="bold")
    ax.margins(0.0)
    ax.set_ylim(ylims)

    return ylims


def draw_ecgs_multi_lead(ecgs, leads_to_show=[0, 1, 2], title="ECG Recording"):
    """Draw multiple ECG leads in a multi-panel plot."""
    fig, axes = plt.subplots(
        len(leads_to_show), 1, figsize=(15, 2.5 * len(leads_to_show))
    )
    if len(leads_to_show) == 1:
        axes = [axes]

    for i, lead in enumerate(leads_to_show):
        draw_ecg(ecgs[0], lead=lead, ax=axes[i])
        axes[i].set_title(f"Lead {lead_names[lead]}", fontweight="bold", pad=10)

    axes[-1].set_xlabel("Time (seconds)", fontweight="bold")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig


def get_ptbxl_ecg_path(ecg_id):
    """Get the file path for a PTB-XL ECG record."""
    return os.path.join(
        "data",
        "ptbxl",
        "records500",
        f"{int(ecg_id / 1000) * 1000:05d}",
        f"{ecg_id:05d}_hr",
    )


def load_and_plot_sample():
    """Load a sample from ECG-QA and create the plot."""
    print("Loading ECG-QA sample...")

    # Load a single sample
    train_dataset = ECGQADataset(split="train", EOS_TOKEN="", max_samples=1)

    if len(train_dataset) == 0:
        print("No samples loaded!")
        return None, None

    sample = train_dataset[0]

    # Get ECG ID and load the raw ECG data
    ecg_ids = sample["ecg_id"]
    if not ecg_ids:
        print("No ECG ID found in sample!")
        return None, None

    ecg_id = ecg_ids[0]  # Use first ECG
    ecg_path = get_ptbxl_ecg_path(ecg_id)

    print(f"Loading ECG {ecg_id} from {ecg_path}")

    try:
        # Load ECG data using wfdb
        ecg_data, meta = wfdb.rdsamp(ecg_path)
        ecgs = [ecg_data.T]  # Transpose to match expected format

        # Create the plot with first 3 leads
        fig = draw_ecgs_multi_lead(
            ecgs,
            leads_to_show=[0, 1, 2],  # I, II, III
            title=f"ECG Recording {ecg_id} - Clinical Question Analysis",
        )

        # Save the plot
        output_path = os.path.join(os.path.dirname(__file__), "ecg_example_real.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"ECG plot saved to: {output_path}")

        return sample, output_path

    except Exception as e:
        print(f"Error loading ECG data: {e}")
        return sample, None


def create_example_prompt_text(sample):
    """Create the example prompt text to show alongside the plot."""
    if not sample:
        return "No sample available"

    # Extract question from raw data since it's not in the processed sample
    try:
        # Load raw data to get the question
        with open("data/ecg_qa/ecgqa/ptbxl/template/train/000000.json") as f:
            raw_data = json.load(f)

        # Find a good example question
        example_sample = None
        for raw_sample in raw_data[:10]:  # Check first 10 samples
            if raw_sample.get("question_type") == "single-verify":
                example_sample = raw_sample
                break

        if not example_sample:
            example_sample = raw_data[0]

        question = example_sample["question"]
        answer = (
            example_sample["answer"][0]
            if isinstance(example_sample["answer"], list)
            else example_sample["answer"]
        )

    except Exception as e:
        print(f"Error loading question: {e}")
        question = "Does this ECG show symptoms of non-diagnostic t abnormalities?"
        answer = "yes"

    prompt_text = f"""## Example ECG Analysis

**Clinical Context:** {sample.get("primary_clinical_context", "Clinical ECG recording for diagnostic analysis")}

**Question:** {question}

**Expected Answer:** {answer}

This example demonstrates how OpenTSLM processes ECG signals on a millimeter paper grid (similar to clinical practice) and answers specific diagnostic questions about cardiac conditions."""

    return prompt_text


def main():
    """Main function to generate the example plot and prompt."""
    print("=== ECG-QA Example Generator ===")

    # Load sample and create plot
    sample, plot_path = load_and_plot_sample()

    if plot_path:
        print("✅ ECG plot generated successfully!")

        # Create example prompt text
        prompt_text = create_example_prompt_text(sample)

        # Save prompt text
        prompt_path = os.path.join(os.path.dirname(__file__), "example_prompt.md")
        with open(prompt_path, "w") as f:
            f.write(prompt_text)

        print(f"✅ Example prompt saved to: {prompt_path}")
        print()
        print("=== Example Prompt ===")
        print(prompt_text)

        return plot_path, prompt_path
    else:
        print("❌ Failed to generate ECG plot")
        return None, None


if __name__ == "__main__":
    main()
