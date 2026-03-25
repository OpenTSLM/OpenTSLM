# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import numpy as np
import pandas as pd

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.prompt.full_prompt import FullPrompt

# 1. Load the model
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")
model = OpenTSLMFlamingo(
    device=device,
    llm_id="google/gemma-2b",  # or whatever you used for training
)

model.load_from_file("../models/best_model.pt")


# 2. Load the M4 series-M42150 from CSV
csv_path = "../data/M4TimeSeriesCaptionDataset/m4_series_Monthly.csv"
df = pd.read_csv(csv_path)
row = df[df["id"] == "series-M42150"].iloc[0]
series_str = row["series"]
# Remove brackets and split
series = [
    float(x)
    for x in series_str.strip("[]").replace("\n", "").replace(" ", "").split(",")
    if x
]
series = np.array(series, dtype=np.float32)
mean = series.mean()
std = series.std()
normalized_series = (series - mean) / std if std > 0 else series - mean

# 3. Build the prompt
pre_prompt = TextPrompt("You are an expert in time series analysis.")
ts_text = f"This is the time series, it has mean {mean:.4f} and std {std:.4f}:"
ts_prompt = TextTimeSeriesPrompt(ts_text, normalized_series.tolist())
post_prompt = TextPrompt(
    "Please generate a detailed caption for this time-series, describing it as accurately as possible."
)

# 4. Build the batch (list of dicts)
prompt = FullPrompt(
    pre_prompt,
    [ts_prompt],
    post_prompt,
)

print(model.eval_prompt(prompt))
