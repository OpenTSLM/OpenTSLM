# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch.nn as nn


class MLPProjector(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()

        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(0.0),
        ).to(device)

    def forward(self, x):
        return self.projector(x)
