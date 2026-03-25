# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from abc import abstractmethod
import torch
import torch.nn as nn

from opentslm.model_config import ENCODER_OUTPUT_DIM


class TimeSeriesEncoderBase(nn.Module):
    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
