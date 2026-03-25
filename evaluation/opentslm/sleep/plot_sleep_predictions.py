#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Plot SleepEDF time series samples from sleep_cot_data.csv.
Each sample is plotted as a PNG with EEG data and the full_prediction as text.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "sleep_cot_data.csv"
OUTPUT_DIR = "sleep_cot_plots"

# Publication style
plt.style.use("seaborn-v0_8")
sns.set_palette("colorblind")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

display_label_map = {
    "W": "Wake",
    "N1": "Non-REM stage 1",
    "N2": "Non-REM stage 2",
    "N3": "Non-REM stage 3",
    "N4": "Non-REM stage 4",
    "REM": "REM sleep",
    "M": "Movement",
    "Unknown": "Unknown",
}


def plot_sample(row, idx):
    eeg_data = np.array(json.loads(row["eeg_data"]))
    full_pred = row["full_prediction"]
    gt_label = row["ground_truth_label"]
    pred_label = row["predicted_label"]
    sample_idx = row["sample_index"]
    series_length = row["series_length"]

    # Map labels to pretty names
    pretty_gt = display_label_map.get(gt_label, gt_label)
    pretty_pred = display_label_map.get(pred_label, pred_label)

    # Normalize text length to exactly 800 characters
    text_length = 900
    if len(full_pred) < text_length:
        # Pad with whitespace if shorter
        full_pred = full_pred + " " * (text_length - len(full_pred))
    elif len(full_pred) > text_length:
        # Truncate if longer
        full_pred = full_pred[:text_length]

    # Add extra newlines to ensure consistent text box height
    full_pred = full_pred + "\n"

    # Normalize EEG data for plotting
    mean = np.mean(eeg_data)
    std = np.std(eeg_data) if np.std(eeg_data) > 0 else 1.0
    eeg_plot = (eeg_data - mean) / std

    fig, ax1 = plt.subplots(figsize=(12, 7))
    t = np.arange(len(eeg_plot))
    # Use the same blue color as PAMAP2 plots (first color from colorblind palette)
    ax1.plot(t, eeg_plot, linewidth=2.5, color="#0173B2", alpha=0.8, label="EEG")
    ax1.set_xlabel("Time Step", fontsize=26)
    ax1.set_ylabel("Normalized EEG Amplitude", fontsize=26)
    ax1.set_title(
        f"Sample {sample_idx} | GT: {pretty_gt} | Pred: {pretty_pred}",
        fontsize=22,
        fontweight="bold",
    )
    ax1.legend(fontsize=13, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", which="major", labelsize=26)
    ax1.set_ylim(-3, 3)
    ax1.set_yticks(np.linspace(-3, 3, 7))

    # Add full_prediction as a text box below the plot (same as PAMAP2)
    plt.gcf().text(
        0.01,
        -0.02,
        f"Prediction:\n{full_pred}",
        fontsize=30,
        ha="left",
        va="top",
        wrap=True,
        bbox=dict(
            boxstyle="round", facecolor="whitesmoke", alpha=0.9, edgecolor="gray"
        ),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    fname = f"sample_{idx+1:03d}_gt_{pretty_gt.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH}")
    for idx, row in df.iterrows():
        plot_sample(row, idx)
    print(f"All plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
