#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Alternative visualization for memory usage.
Shows one subplot per base model (Llama, Gemma, etc).
Within each subplot: bars for SoftPrompt vs Flamingo across datasets.
Styled to match the paper-style plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def parse_model_name(llm_id, model_type):
    if llm_id.startswith("meta-llama/"):
        base_name = llm_id.replace("meta-llama/", "")
    elif llm_id.startswith("google/"):
        base_name = llm_id.replace("google/", "")
    else:
        base_name = llm_id

    # Normalize base model names to match expected order
    if "Llama-3.2-1B" in base_name:
        base_name = "Llama-3.2-1B"
    elif "Llama-3.2-3B" in base_name:
        base_name = "Llama-3.2-3B"
    elif "gemma-3-270m" in base_name:
        base_name = "Gemma-3-270M"
    elif "gemma-3-1b-pt" in base_name:
        base_name = "Gemma-3-1B-pt"

    if model_type == "OpenTSLMSP":
        type_name = "SoftPrompt"
    elif model_type == "OpenTSLMFlamingo":
        type_name = "Flamingo"
    else:
        type_name = model_type

    return base_name, type_name


def plot_memory_usage(csv_file="merged_success.csv"):
    # --- Publication style settings ---
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "Times New Roman", "DejaVu Serif"],
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "legend.fontsize": 17,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "0.15",
        }
    )

    df = pd.read_csv(csv_file)

    # Replace -1 with 50GB placeholder
    df["peak_cuda_reserved_gb"] = df["peak_cuda_reserved_gb"].replace(-1, 50.0)

    # Parse into base + config
    df[["base_model", "config"]] = df.apply(
        lambda row: pd.Series(parse_model_name(row["llm_id"], row["model"])), axis=1
    )

    # Order datasets
    dataset_order = ["TSQA", "HAR-CoT", "SleepEDF-CoT", "ECG-QA-CoT"]
    df["dataset"] = pd.Categorical(
        df["dataset"], categories=dataset_order, ordered=True
    )

    # Order configs
    config_order = ["SoftPrompt", "Flamingo"]
    df["config"] = pd.Categorical(df["config"], categories=config_order, ordered=True)

    # Define consistent order for base models
    model_order = ["Gemma-3-270M", "Gemma-3-1B-pt", "Llama-3.2-1B", "Llama-3.2-3B"]

    # Get base models in the specified order
    base_models = [bm for bm in model_order if bm in df["base_model"].unique()]

    # Plot
    n_models = len(base_models)
    fig, axes = plt.subplots(1, n_models, figsize=(3.2 * n_models, 3.5), sharey=True)

    if n_models == 1:
        axes = [axes]

    # Muted, paper-friendly palette
    palette = {
        "SoftPrompt": "#4477AA",  # muted blue
        "Flamingo": "#CC6677",  # muted red
    }

    for ax, base_model in zip(axes, base_models):
        subdf = df[df["base_model"] == base_model]

        sns.barplot(
            data=subdf,
            x="dataset",
            y="peak_cuda_reserved_gb",
            hue="config",
            ax=ax,
            palette=palette,
            edgecolor="0.3",
            linewidth=0.6,
            dodge=True,
            width=0.8,
            legend=(ax == axes[0]),  # Show legend only in first subplot
        )

        ax.set_title(base_model, fontsize=19, fontweight="bold")
        ax.set_xlabel("", fontsize=18, fontweight="bold")
        ax.set_ylabel("Peak VRAM Usage (GB)", fontsize=18, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.set_facecolor("#F8F9FA")
        ax.grid(axis="y", alpha=0.5, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 110)

        # Style the legend in the first subplot
        if ax == axes[0]:
            legend = ax.get_legend()
            if legend:
                legend.set_title(None)  # Remove "config" title
                legend.get_frame().set_facecolor("white")
                legend.get_frame().set_edgecolor("0.3")
                legend.get_frame().set_linewidth(1.0)
                legend.get_frame().set_alpha(1.0)
                legend.set_frame_on(True)

        # Add value labels with larger font and rotation
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", padding=2, fontsize=15, rotation=90)

    plt.tight_layout(pad=0.5)

    for fmt in ["png", "pdf"]:
        plt.savefig(
            f"memory_usage_facet.{fmt}",
            dpi=300 if fmt == "png" else None,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="white",
            format=fmt,
        )
    plt.show()


if __name__ == "__main__":
    plot_memory_usage()
