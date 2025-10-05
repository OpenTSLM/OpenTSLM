#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn


from src.model_config import TRANSFORMER_INPUT_DIM, ENCODER_OUTPUT_DIM, PATCH_SIZE
from model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class TransformerMLPEncoder(TimeSeriesEncoderBase):
    def __init__(
        self,
        input_dim: int = TRANSFORMER_INPUT_DIM,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        num_heads: int = 8,
        num_layers: int = 6,
        patch_size: int = PATCH_SIZE,
        ff_dim: int = 2048,
        max_patches: int = 1024,
    ):
        """
        Args:
            embed_dim: dimension of patch embeddings
            num_heads: number of attention heads
            num_layers: number of TransformerEncoder layers
            patch_size: length of each patch
            ff_dim: hidden size of the feed‐forward network inside each encoder layer
            dropout: dropout probability
            max_patches: maximum number of patches expected per sequence (for pos emb)
        """
        super().__init__(input_dim, output_dim, dropout)
        self.patch_size = patch_size

        if input_dim % patch_size != 0:
            raise RuntimeError(
                "transformer encoder input dim must be divisible by patch size"
            )
        transformer_input_size = self.input_dim // patch_size
        self.patch_embed = nn.Linear(self.input_dim, transformer_input_size)

        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, self.input_dim))

        # 3) Input norm + dropout
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_dropout = nn.Dropout(self.dropout)

        # 4) Stack of TransformerEncoder layers with higher ff_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape [B, L], a batch of raw time series.
        Returns:
            FloatTensor of shape [B, N, embed_dim], where N = L // patch_size.
        """
        B, L = x.shape
        if L % self.patch_size != 0:
            raise ValueError(
                f"Sequence length {L} not divisible by patch_size {self.patch_size}"
            )

        # reshape to (B, 1, L)
        x = x.unsqueeze(1)

        # linear patch embed
        x = self.patch_embed(x)

        # transpose to (B, N, embed_dim)
        x = x.transpose(1, 2)

        # add positional embeddings (truncate or expand as needed)
        N = x.size(1)
        if N > self.pos_embed.size(1):
            raise ValueError(
                f"Time series of length {N*4} is too long; max supported is {self.pos_embed.size(1)*4}. Change max_patches parameter in {__file__}"
            )
        pos = self.pos_embed[:, :N, :]
        x = x + pos

        # norm + dropout
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # apply Transformer encoder
        x = self.encoder(x)

        return x
