#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Plot memory usage on simulation datasets from memory_simulation.csv.

- Only uses datasets starting with 'Simulation-'.
- Extracts time series length (L) and number of series (N).
- Computes total_length = N * L.
- Plots memory vs total_length per base model, comparing SoftPrompt vs Flamingo.
- OOM runs (> 180GB) are shown with a dashed line, red X, and "OOM" label.
- Always shows panels in order: gemma-270m, gemma-1b, llama-1b, llama-3b.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import re

OOM_THRESHOLD = 180  # GB


def parse_model_name(llm_id, model_type):
    """Return base_model, config (SoftPrompt or Flamingo)."""
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


def parse_simulation_dataset(name):
    """Parse Simulation dataset name like 'Simulation-L10-N5' → (L=10, N=5)."""
    match = re.match(r"Simulation-L(\d+)-N(\d+)", name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def plot_memory_usage_sim(csv_file="memory_simulation.csv"):
    # --- Paper-style settings ---
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

    # Replace -1 with NaN (ignore failed runs)
    df["peak_cuda_reserved_gb"] = df["peak_cuda_reserved_gb"].replace(-1, pd.NA)

    # Keep only simulation datasets
    df = df[df["dataset"].str.startswith("Simulation-")]

    # Parse model name and dataset details
    df[["base_model", "config"]] = df.apply(
        lambda row: pd.Series(parse_model_name(row["llm_id"], row["model"])), axis=1
    )
    df[["L", "N"]] = df["dataset"].apply(
        lambda s: pd.Series(parse_simulation_dataset(s))
    )
    df = df.dropna(subset=["L", "N"])
    df["L"] = df["L"].astype(int)
    df["N"] = df["N"].astype(int)

    # Compute total sequence length
    df["total_length"] = df["L"] * df["N"]

    # Sort
    df = df.sort_values(by=["base_model", "config", "total_length"])

    # Fixed base_model order
    base_model_order = ["Gemma-3-270M", "Gemma-3-1B-pt", "Llama-3.2-1B", "Llama-3.2-3B"]

    # One subplot per model (always 4)
    n_models = len(base_model_order)
    fig, axes = plt.subplots(1, n_models, figsize=(3.2 * n_models, 3.2), sharey=True)

    if n_models == 1:
        axes = [axes]

    # Muted palette for configs - order matters for legend
    palette = {"SoftPrompt": "#4477AA", "Flamingo": "#CC6677"}
    config_order = ["SoftPrompt", "Flamingo"]

    for ax, base_model in zip(axes, base_model_order):
        subdf = df[df["base_model"] == base_model]

        if subdf.empty:
            ax.set_title(base_model, fontsize=13, fontweight="bold")
            ax.set_facecolor("#F8F9FA")
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", fontsize=10, color="gray"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        for cfg in config_order:
            cfg_df = subdf[subdf["config"] == cfg]
            if cfg_df.empty:
                continue
            cfg_df = cfg_df.sort_values("total_length")
            color = palette[cfg]

            # Successful runs (≤ threshold)
            ok_df = cfg_df[cfg_df["peak_cuda_reserved_gb"] <= OOM_THRESHOLD]
            ax.plot(
                ok_df["total_length"],
                ok_df["peak_cuda_reserved_gb"],
                label=cfg,
                color=color,
                linewidth=4.0,
                alpha=0.9,
            )

            # First OOM run (if any)
            oom_df = cfg_df[cfg_df["peak_cuda_reserved_gb"] > OOM_THRESHOLD]
            if not oom_df.empty and not ok_df.empty:
                first_oom = oom_df.iloc[0]
                last_ok = ok_df.iloc[-1]

                # dashed line up to OOM
                ax.plot(
                    [last_ok["total_length"], first_oom["total_length"]],
                    [last_ok["peak_cuda_reserved_gb"], OOM_THRESHOLD * 1.05],
                    color=color,
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.8,
                )

                # red X marker
                ax.scatter(
                    first_oom["total_length"],
                    OOM_THRESHOLD * 1.05,
                    color="red",
                    marker="x",
                    s=80,
                    linewidth=3,
                    zorder=5,
                )
                ax.text(
                    first_oom["total_length"],
                    OOM_THRESHOLD * 1.05,
                    "OOM",
                    color="red",
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                )

        # Titles & labels
        ax.set_title(base_model, fontsize=19, fontweight="bold")

        # Only show axis labels on specific subplots
        if ax == axes[0]:  # Leftmost subplot
            ax.set_ylabel("Peak VRAM Usage (GB)", fontsize=18, fontweight="bold")
            ax.set_xlabel(
                "Total Sequence Length (N × L)", fontsize=18, fontweight="bold"
            )
        else:
            ax.set_ylabel("")
            ax.set_xlabel("")
        ax.set_facecolor("#F8F9FA")
        ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
        ax.minorticks_on()
        ax.tick_params(axis="both", labelsize=17)

        # Legend only in first subplot
        if ax == axes[0]:
            leg = ax.legend(
                title=None,
                fontsize=17,
                loc="best",
                frameon=True,
                framealpha=0.95,
                edgecolor="0.3",
            )
            for text in leg.get_texts():
                text.set_fontweight("bold")

    plt.tight_layout(pad=0.5)
    for fmt in ["png", "pdf"]:
        plt.savefig(
            f"memory_usage_simulation.{fmt}",
            dpi=300 if fmt == "png" else None,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="white",
            format=fmt,
        )
    plt.show()


if __name__ == "__main__":
    plot_memory_usage_sim()
