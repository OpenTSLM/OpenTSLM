#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Alternative visualization for memory usage.
Shows one subplot per base model (Llama, Gemma, etc).
Within each subplot: bars for SoftPrompt vs Flamingo across datasets.
Uses clean, colorblind-friendly colors.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_model_name(llm_id, model_type):
    if llm_id.startswith("meta-llama/"):
        base_name = llm_id.replace("meta-llama/", "")
    elif llm_id.startswith("google/"):
        base_name = llm_id.replace("google/", "")
    else:
        base_name = llm_id

    if model_type == "OpenTSLMSP":
        type_name = "SoftPrompt"
    elif model_type == "OpenTSLMFlamingo":
        type_name = "Flamingo"
    else:
        type_name = model_type

    return base_name, type_name


def plot_memory_usage(csv_file="memory_use.csv"):
    df = pd.read_csv(csv_file)

    # Replace -1 with 50GB
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

    # Get unique base models
    base_models = df["base_model"].unique()

    # Plot
    n_models = len(base_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)

    if n_models == 1:
        axes = [axes]

    # Colorblind-friendly palette
    palette = {"SoftPrompt": "#4C78A8", "Flamingo": "#F58518"}

    for ax, base_model in zip(axes, base_models):
        subdf = df[df["base_model"] == base_model]

        sns.barplot(
            data=subdf,
            x="dataset",
            y="peak_cuda_reserved_gb",
            hue="config",
            ax=ax,
            palette=palette,
        )

        ax.set_title(base_model, fontsize=14, fontweight="bold")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Peak CUDA Reserved GB")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", padding=2, fontsize=8)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        title="Config",
        fontsize=10,
    )
    fig.suptitle("Peak CUDA Reserved Memory by Model, Dataset, and Config", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(
        "memory_usage_facet.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.show()


if __name__ == "__main__":
    plot_memory_usage()
