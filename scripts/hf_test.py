# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
from opentslm import OpenTSLM
from opentslm.prompt import TextPrompt, TextTimeSeriesPrompt, FullPrompt

# Load model
model = OpenTSLM.load_pretrained("OpenTSLM/gemma-3-270m-pt-har-flamingo")

# Create prompt with raw time series data (normalization handled automatically)
prompt = FullPrompt(
    pre_prompt=TextPrompt("You are an expert in HAR analysis."),
    text_time_series_prompt_list=[
        TextTimeSeriesPrompt("X-axis accelerometer", [2.34, 2.34, 7.657, 3.21, -1.2])
    ],
    post_prompt=TextPrompt("What activity is this? Reasn step by step providing a full rationale before replying.")
)

# Generate response
output = model.eval_prompt(prompt, normalize=True)
print(output)
