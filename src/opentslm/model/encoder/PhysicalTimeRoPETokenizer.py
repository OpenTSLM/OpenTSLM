# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math
from collections.abc import Sequence

import torch
from torch import nn

from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase
from opentslm.model_config import ENCODER_OUTPUT_DIM


class PhysicalTimeRoPETokenizer(TimeSeriesEncoderBase):
    """Project observations to tokens and rotate feature pairs by elapsed time.

    Each observation is mapped to ``output_dim`` features by a linear layer, and
    feature pair ``k`` is rotated by ``2 * pi * ((t - t_first) mod P_k) / P_k``,
    where ``t_first`` is the first valid timestamp in each series.

    Args:
        output_dim: Even token embedding dimension.
        dropout: Dropout probability applied after rotary embedding.
        input_channels: Number of channels per observation.
        min_period: Shortest log-spaced period (``2.0`` is the Nyquist period for unit steps).
        max_period: Longest log-spaced period, in the same time unit as ``timestamps``
            (default ``365.25`` corresponds to one Julian year when timestamps are in days).
        pinned_periods: Explicit positive periods placed at the front of the period bank
            before filling remaining feature pairs log-uniformly between ``min_period``
            and ``max_period``.

    Timestamps and periods share one user-chosen unit. When ``timestamps`` is omitted,
    consecutive valid observations advance by one unit. Pass ``float64`` timestamps when
    clock offsets are large relative to inter-sample gaps. On Apple MPS, ``float64``
    modular phase reduction runs on CPU before transferring rotation coefficients to MPS.
    """

    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        *,
        input_channels: int = 1,
        min_period: float = 2.0,
        max_period: float = 365.25,
        pinned_periods: Sequence[float] = (),
    ):
        super().__init__(output_dim, dropout)
        if output_dim < 2 or output_dim % 2 != 0 or input_channels < 1:
            raise ValueError(
                f"output_dim must be positive and even, and input_channels positive "
                f"(got output_dim={output_dim}, input_channels={input_channels})"
            )
        if not (0 < min_period <= max_period < math.inf):
            raise ValueError(f"period bounds must be finite, positive and ordered, got ({min_period}, {max_period})")
        pins = tuple(float(period) for period in pinned_periods)
        pairs = output_dim // 2
        if len(pins) > pairs or any(not (0 < period < math.inf) for period in pins):
            raise ValueError("supply at most output_dim / 2 finite, positive pinned periods")
        remaining = pairs - len(pins)
        log_min, log_max = math.log(min_period), math.log(max_period)
        self.periods = pins + tuple(
            math.exp(log_min + (log_max - log_min) * index / max(remaining - 1, 1)) for index in range(remaining)
        )
        self.input_channels = input_channels
        self.projection = nn.Linear(input_channels, output_dim)
        self.input_dropout = nn.Dropout(dropout)

    def extra_repr(self) -> str:
        return (
            f"input_channels={self.input_channels}, output_dim={self.output_dim}, "
            f"dropout={self.dropout}, num_periods={len(self.periods)}, "
            f"period_range=({min(self.periods):.6g}, {max(self.periods):.6g})"
        )

    def forward(
        self,
        values: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[B, L, output_dim]`` tokens with zeros at masked positions."""
        return self.encode_with_mask(values, timestamps, mask)[0]

    def encode_with_mask(
        self,
        values: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[B, L, output_dim]`` tokens and a ``[B, L]`` boolean validity mask (``True`` = valid).

        Downstream mask conventions:
        - Mean pooling: ``tokens.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)``.
        - ``nn.MultiheadAttention``: pass ``key_padding_mask=~valid`` (``True`` ignores padding)
          and guard rows with no valid observations (``~valid.any(dim=1)``).
        - ``F.scaled_dot_product_attention``: pass ``attn_mask=valid[:, None, None, :]``
          (``True`` attends to valid keys) and guard rows with no valid observations.
        - Unmasked resamplers (such as ``PerceiverResampler``): select ``tokens[row, valid[row]]``
          for each non-empty series before resampling.
        """
        raw_shape = tuple(values.shape)
        # When L == input_channels in 3D input, [B, L, C] takes precedence over [B, C, L].
        if values.ndim == 2 and self.input_channels == 1:
            values = values.unsqueeze(-1)
        elif values.ndim == 3 and values.shape[-1] != self.input_channels and values.shape[1] == self.input_channels:
            values = values.transpose(1, 2)
        if values.ndim != 3 or values.shape[-1] != self.input_channels:
            expected = (
                "[B, L], [B, L, C] or [B, C, L]"
                if self.input_channels == 1
                else f"[B, L, {self.input_channels}] or [B, {self.input_channels}, L]"
            )
            raise ValueError(f"values must have shape {expected}, got {raw_shape}")

        shape = values.shape[:2]
        if mask is None:
            valid = torch.ones(shape, dtype=torch.bool, device=values.device)
        else:
            if mask.shape != shape or mask.dtype != torch.bool:
                raise ValueError(
                    f"mask must be boolean with shape {tuple(shape)}, got {mask.dtype} {tuple(mask.shape)}"
                )
            valid = mask.to(device=values.device)

        clock_device = torch.device("cpu") if values.device.type == "mps" else values.device
        valid_clock = valid.to(device=clock_device)
        if timestamps is None:
            elapsed = (valid_clock.cumsum(dim=-1) - 1).to(dtype=torch.float64)
        else:
            if timestamps.shape != shape or timestamps.dtype == torch.bool or timestamps.is_complex():
                raise ValueError(
                    f"timestamps must be real-valued with shape {tuple(shape)}, "
                    f"got {timestamps.dtype} {tuple(timestamps.shape)}"
                )
            # Move to clock_device before promoting to float64 so MPS autograd never allocates float64 on MPS.
            times = timestamps.to(device=clock_device).to(dtype=torch.float64)
            times = torch.where(valid_clock, times, 0.0)
            elapsed = times
            if shape[1]:
                first_index = valid_clock.to(torch.long).argmax(dim=-1, keepdim=True)
                elapsed = times - times.gather(1, first_index)
            elapsed = torch.where(valid_clock, elapsed, 0.0)
            if not torch.isfinite(elapsed).all():
                raise ValueError("valid timestamps and their elapsed differences must be finite")

        if mask is None:
            projected = self.projection(values)
            return self.input_dropout(self._rotate(projected, elapsed)), valid
        # Compact valid observations before projection so masked values (including NaN/Inf)
        # cannot affect tokens, batch statistics, or gradients on padded slots.
        projected = self.projection(values[valid])
        rotated = self.input_dropout(self._rotate(projected, elapsed[valid_clock]))
        tokens = rotated.new_zeros((*shape, self.output_dim))
        tokens[valid] = rotated
        return tokens, valid

    def _rotate(self, projected: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        # Keep the period bank in Python so module-wide dtype conversions
        # cannot round the periods before the float64 modular reduction.
        clock_device = torch.device("cpu") if projected.device.type == "mps" else projected.device
        with torch.autocast(device_type=projected.device.type, enabled=False):
            periods = torch.tensor(self.periods, dtype=torch.float64, device=clock_device)
            elapsed = elapsed.to(device=clock_device).to(dtype=torch.float64).unsqueeze(-1)
            phase = (2.0 * math.pi) * (torch.remainder(elapsed, periods) / periods)
            rotation_dtype = torch.promote_types(projected.dtype, torch.float32)
            cos_phase = phase.cos().to(dtype=rotation_dtype).to(device=projected.device)
            sin_phase = phase.sin().to(dtype=rotation_dtype).to(device=projected.device)
            real, imag = projected.to(rotation_dtype).unflatten(-1, (-1, 2)).unbind(dim=-1)
            rotated = torch.stack(
                (real * cos_phase - imag * sin_phase, real * sin_phase + imag * cos_phase),
                dim=-1,
            ).flatten(-2)
        return rotated.to(dtype=projected.dtype)
