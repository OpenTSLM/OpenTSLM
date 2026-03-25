#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Paper-style plots: memory usage scaling with N for different lengths (L).

- Rows = config (SoftPrompt, Flamingo)
- Cols = sequence lengths (L) [excluding L=1]
- Hue = base model
- Y-axis sharing logic:
    * Flamingo: all panels share y-axis
    * SoftPrompt: all panels have independent y-axes
- OOM cases (peak_cuda_reserved_gb > OOM_THRESHOLD) are shown by extending the
  line upward and marking with a red X + "OOM".
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import re
from matplotlib.lines import Line2D

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


def plot_memory_usage_paper(csv_file="memory_simulation.csv"):
    # Publication style
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

    # Load & preprocess
    df = pd.read_csv(csv_file)
    df["peak_cuda_reserved_gb"] = df["peak_cuda_reserved_gb"].replace(-1, pd.NA)
    df = df[df["dataset"].str.startswith("Simulation-")]
    df[["base_model", "config"]] = df.apply(
        lambda row: pd.Series(parse_model_name(row["llm_id"], row["model"])), axis=1
    )
    df[["L", "N"]] = df["dataset"].apply(
        lambda s: pd.Series(parse_simulation_dataset(s))
    )
    df = df.dropna(subset=["L", "N"])
    df["L"] = df["L"].astype(int)
    df["N"] = df["N"].astype(int)
    df = df[df["L"] != 1]
    df = df.sort_values(by=["base_model", "config", "L", "N"])

    # Palette + markers - use consistent colors with the other script
    # Define consistent order and colors for each model
    model_order = ["Gemma-3-270M", "Gemma-3-1B-pt", "Llama-3.2-1B", "Llama-3.2-3B"]
    color_map = {
        "Gemma-3-270M": "#4477AA",
        "Gemma-3-1B-pt": "#66CCEE",
        "Llama-3.2-1B": "#228833",
        "Llama-3.2-3B": "#CC6677",
    }

    # Get base models in the specified order
    base_models = [bm for bm in model_order if bm in df["base_model"].unique()]
    custom_palette = [color_map.get(bm, "#888888") for bm in base_models]
    markers_dict = dict(zip(base_models, ["o", "s", "^", "D", "p", "X", "*"]))

    # Unique sequence lengths
    unique_L = sorted(df["L"].unique())

    # Create subplot grid manually: 2 rows (SoftPrompt, Flamingo)
    fig, axes = plt.subplots(
        2,
        len(unique_L),
        figsize=(3.2 * len(unique_L), 6),
        sharex="col",
    )

    # Row mapping
    row_map = {"SoftPrompt": 0, "Flamingo": 1}

    # Precompute Flamingo y-lims
    flamingo_df = df[df["config"] == "Flamingo"]
    flamingo_ymin, flamingo_ymax = None, None
    if not flamingo_df.empty:
        flamingo_ymin = flamingo_df["peak_cuda_reserved_gb"].min(skipna=True)
        flamingo_ymax = flamingo_df["peak_cuda_reserved_gb"].max(skipna=True)

    flamingo_ymin = 0
    flamingo_ymax = max(flamingo_ymax if flamingo_ymax else 0, OOM_THRESHOLD * 1.1)

    # Iterate configs
    for cfg in ["SoftPrompt", "Flamingo"]:
        cfg_df = df[df["config"] == cfg]

        for j, L in enumerate(unique_L):
            ax = axes[row_map[cfg], j]
            subdf = cfg_df[cfg_df["L"] == L]

            for bm, sdf in subdf.groupby("base_model"):
                sdf = sdf.sort_values("N")

                # Split successful vs OOM runs
                ok_df = sdf[sdf["peak_cuda_reserved_gb"] <= OOM_THRESHOLD]
                oom_df = sdf[sdf["peak_cuda_reserved_gb"] > OOM_THRESHOLD]

                # Normal line
                if not ok_df.empty:
                    ax.plot(
                        ok_df["N"],
                        ok_df["peak_cuda_reserved_gb"],
                        label=bm,
                        color=custom_palette[base_models.index(bm)],
                        marker=markers_dict[bm],
                        linewidth=2.2,
                        markersize=5,
                        alpha=0.9,
                    )

                # OOM handling
                if not oom_df.empty:
                    first_oom = oom_df.iloc[0]
                    last_ok_y = (
                        ok_df["peak_cuda_reserved_gb"].iloc[-1]
                        if not ok_df.empty
                        else OOM_THRESHOLD * 0.9
                    )

                    # extend line upward
                    ax.plot(
                        [
                            ok_df["N"].iloc[-1] if not ok_df.empty else first_oom["N"],
                            first_oom["N"],
                        ],
                        [last_ok_y, OOM_THRESHOLD * 1.05],
                        color=custom_palette[base_models.index(bm)],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.8,
                    )

                    # red X marker at OOM
                    ax.scatter(
                        first_oom["N"],
                        OOM_THRESHOLD * 1.05,
                        color="red",
                        marker="x",
                        s=70,
                        linewidth=2,
                        zorder=5,
                    )
                    ax.text(
                        first_oom["N"],
                        OOM_THRESHOLD * 1.05,
                        "OOM",
                        color="red",
                        fontsize=9,
                        fontweight="bold",
                        ha="center",
                        va="bottom",
                    )

            # Titles
            if row_map[cfg] == 0:
                ax.set_title(f"L = {L}", fontsize=19, fontweight="bold")

            # Y labels only leftmost col
            if j == 0:
                ax.set_ylabel(f"{cfg}", fontsize=18, fontweight="bold")
            else:
                ax.set_ylabel("")

            # Styling
            ax.set_facecolor("#F8F9FA")
            ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.5)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
            ax.minorticks_on()
            # Remove individual x-axis labels - will add global one later
            ax.set_xlabel("")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.tick_params(axis="x", which="both", labelbottom=True)

            # Y-axis rules
            if cfg == "Flamingo":
                ax.set_ylim(0, 65)
            else:
                if j <= 2:
                    ax.set_ylim(0, 25)
                elif j == 3:
                    ax.set_ylim(0, 200)

    # Legend (global, right side) - create in specified order
    # Get handles and labels from first subplot
    handles_dict = {}
    labels_dict = {}
    for handle, label in zip(*axes[0, 0].get_legend_handles_labels()):
        handles_dict[label] = handle
        labels_dict[label] = label

    # Create ordered handles and labels
    ordered_handles = []
    ordered_labels = []
    for bm in base_models:
        if bm in handles_dict:
            ordered_handles.append(handles_dict[bm])
            ordered_labels.append(labels_dict[bm])

    # Add OOM handle
    oom_handle = Line2D(
        [0],
        [0],
        color="red",
        marker="x",
        linestyle="--",
        markersize=8,
        label="Out of Memory (OOM)",
    )
    ordered_handles.append(oom_handle)
    ordered_labels.append("Out of Memory (OOM)")

    # Legend in top left plot only
    top_left_ax = axes[0, 0]
    top_left_ax.legend(
        ordered_handles,
        ordered_labels,
        title=None,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="0.3",
        fontsize=12,
    )

    # Add main vertical title
    fig.text(
        0.02,
        0.5,
        "Peak Memory (GB)",
        rotation=90,
        fontsize=20,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Add global x-axis label spanning the bottom row
    fig.text(
        0.5,
        0.01,
        "Number of Time Series (N)",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Layout - no longer need space for bottom legend
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.08)

    # Save
    for fmt in ["png", "pdf"]:
        plt.savefig(
            f"memory_usage_paper.{fmt}",
            dpi=300 if fmt == "png" else None,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="white",
            format=fmt,
        )
    plt.show()


if __name__ == "__main__":
    plot_memory_usage_paper()
