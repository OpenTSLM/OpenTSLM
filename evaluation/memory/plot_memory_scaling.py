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
- OOM cases (status != "ok" or missing memory) shown as red X markers
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import re
from matplotlib.lines import Line2D


def parse_model_name(llm_id, model_type):
    """Return base_model, config (SoftPrompt or Flamingo)."""
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


def parse_simulation_dataset(name):
    """Parse Simulation dataset name like 'Simulation-L10-N5' → (L=10, N=5)."""
    match = re.match(r"Simulation-L(\d+)-N(\d+)", name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def plot_memory_usage_paper(csv_file="memory_simulation.csv"):
    # Publication style
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino", "Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "0.15",
    })

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

    # Palette + markers
    base_models = list(df["base_model"].unique())
    custom_palette = sns.color_palette("tab10", n_colors=len(base_models))
    markers_dict = dict(zip(
        base_models,
        ["o", "s", "^", "D", "p", "X", "*"]
    ))

    # Unique sequence lengths
    unique_L = sorted(df["L"].unique())

    # Create subplot grid manually: 2 rows (SoftPrompt, Flamingo)
    fig, axes = plt.subplots(
        2, len(unique_L),
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
    flamingo_ymax = 65

    # Iterate configs
    for cfg in ["SoftPrompt", "Flamingo"]:
        cfg_df = df[df["config"] == cfg]

        for j, L in enumerate(unique_L):
            ax = axes[row_map[cfg], j]
            subdf = cfg_df[cfg_df["L"] == L]

            ymax_local = subdf["peak_cuda_reserved_gb"].max(skipna=True)
            oom_y = (ymax_local if pd.notna(ymax_local) else 0) * 1.05 + 5

            for bm, sdf in subdf.groupby("base_model"):
                sdf = sdf.sort_values("N")

                # Successful runs
                ok_df = sdf[(sdf.get("status", "ok") == "ok") & sdf["]()]()_
