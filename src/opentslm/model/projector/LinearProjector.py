# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from jaxtyping import Float


class LinearProjector(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim).to(device)

    def forward(
        self, x: Float[torch.Tensor, "*batch patches encoder_dim"]
    ) -> Float[torch.Tensor, "*batch patches llm_dim"]:
        return self.projector(x)
