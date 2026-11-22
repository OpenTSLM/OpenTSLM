# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from opentslm.model.encoder.PhysicalTimeRoPETokenizer import PhysicalTimeRoPETokenizer
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase

# Arbitrary incommensurate periods, including values below the default min_period of 2.0.
PERIODS = (0.5, 0.7, 1.1, 2.3, 3.7, 11.3)


def test_lazy_package_and_encoder_imports():
    """Importing opentslm or PhysicalTimeRoPETokenizer must not load LLM backends."""
    code = (
        "import sys; import opentslm; "
        "assert 'OpenTSLM' in dir(opentslm); "
        "from opentslm.model.encoder import PhysicalTimeRoPETokenizer as A; "
        "from opentslm.model.encoder.PhysicalTimeRoPETokenizer import PhysicalTimeRoPETokenizer as B; "
        "assert A is B; "
        "assert 'transformers' not in sys.modules; "
        "assert 'open_flamingo' not in sys.modules; "
        "assert 'peft' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_base_contract_and_uniform_step():
    """Verify base-class inheritance, state_dict keys, and accepted channel layouts."""
    encoder = PhysicalTimeRoPETokenizer(output_dim=12).eval()
    assert isinstance(encoder, TimeSeriesEncoderBase)
    assert set(encoder.state_dict().keys()) == {"projection.weight", "projection.bias"}
    assert "num_periods=6" in repr(encoder)

    values = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    times = torch.arange(8, dtype=torch.float64).expand_as(values)
    tokens = encoder(values)
    assert tokens.shape == (2, 8, 12)
    assert torch.equal(tokens, encoder(values.unsqueeze(1)))
    assert torch.equal(tokens, encoder(values.unsqueeze(-1), times))

    multichannel = PhysicalTimeRoPETokenizer(input_channels=3, output_dim=12).eval()
    wide = values.unsqueeze(-1).expand(-1, -1, 3).contiguous()
    assert torch.equal(multichannel(wide), multichannel(wide.transpose(1, 2)))

    # When L == C, [B, L, C] takes precedence over [B, C, L].
    square = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    expected_square = multichannel._rotate(
        multichannel.projection(square),
        torch.arange(3, dtype=torch.float64).expand(2, 3),
    )
    assert torch.equal(multichannel(square), expected_square)


def test_implicit_clock_skips_masked_slots():
    """Without explicit timestamps, valid samples advance by one step while skipping padding."""
    encoder = PhysicalTimeRoPETokenizer(output_dim=2, pinned_periods=(4.0,)).eval()
    with torch.no_grad():
        encoder.projection.weight.zero_()
        encoder.projection.bias.copy_(torch.tensor([1.0, 0.0]))
    values = torch.tensor([[9.0, float("nan"), 9.0, float("inf"), 9.0]])
    valid = torch.tensor([[True, False, True, False, True]])
    tokens, returned = encoder.encode_with_mask(values, None, valid)
    assert torch.equal(returned, valid)
    assert torch.count_nonzero(tokens[~valid]) == 0
    expected_valid = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    torch.testing.assert_close(tokens[valid].unsqueeze(0), expected_valid, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(tokens[valid].unsqueeze(0), encoder(torch.ones(1, 3)))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("autocast", [False, True])
def test_half_precision_against_modular_oracle(dtype, autocast):
    """Check bit-exact bf16 and fp16 outputs against a float64-phase, float32-rotation reference."""
    torch.manual_seed(170)
    encoder = PhysicalTimeRoPETokenizer(output_dim=16, pinned_periods=PERIODS).eval()
    assert encoder.periods[: len(PERIODS)] == PERIODS
    values = torch.randn(2, 32)
    times = torch.linspace(0.123, 59.9, 32, dtype=torch.float64).expand_as(values)
    if not autocast:
        encoder = encoder.to(dtype=dtype)
        values = values.to(dtype=dtype)
    with torch.autocast("cpu", dtype=dtype, enabled=autocast):
        projected = encoder.projection(values.unsqueeze(-1))
        actual = encoder(values, times)
    periods = torch.tensor(encoder.periods, dtype=torch.float64)
    angles = (2 * math.pi) * (torch.remainder((times - times[..., :1])[..., None], periods) / periods)
    angles_cos = angles.cos().float()
    angles_sin = angles.sin().float()
    expected = torch.empty_like(projected, dtype=torch.float32)
    x, y = projected[..., ::2].float(), projected[..., 1::2].float()
    expected[..., ::2] = x * angles_cos - y * angles_sin
    expected[..., 1::2] = x * angles_sin + y * angles_cos
    assert actual.dtype == dtype
    assert torch.equal(actual.view(torch.int16), expected.to(dtype=dtype).view(torch.int16))


def test_global_clock_shift_invariance():
    """Verify invariance to global clock shifts across positive and negative epochs."""
    encoder = PhysicalTimeRoPETokenizer(output_dim=12, pinned_periods=PERIODS).eval()
    times = torch.tensor([[0.17, 0.911, 3.3, 20.542], [2.50, 2.625, 4.125, 8.085]], dtype=torch.float64)
    values = torch.tensor([[1.0, 2.0, -1.0, 0.5], [3.0, 5.0, 7.0, -2.0]])
    assert torch.equal(encoder(values, times), encoder(values, times + 1024.0))
    torch.testing.assert_close(encoder(values, times), encoder(values, times + 20_000.0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(encoder(values, times), encoder(values, times - 20_000.0), atol=1e-6, rtol=1e-6)
    # Across ~1e9 cycles of period 0.5, binary64 input representation shifts phase by ~4e-6 rad.
    torch.testing.assert_close(encoder(values, times), encoder(values, times + 1.7e9), atol=1e-5, rtol=1e-5)


def test_mask_removes_padding_from_features_pooling_attention_and_gradients():
    """Verify masked compaction, MHA vs SDPA polarity, dropout zeros, and analytical time gradients."""
    torch.manual_seed(23)
    encoder = PhysicalTimeRoPETokenizer(input_channels=2, output_dim=12).eval()
    values = torch.tensor(
        [[[float("nan"), float("inf")], [2.0, 8.0], [float("inf"), 77.0], [3.0, 6.0], [4.0, 4.0], [999.0, 123.0]]],
        requires_grad=True,
    )
    times = torch.tensor(
        [[float("nan"), 100.0, -12345.0, 102.25, 109.0, 150.0]], dtype=torch.float64, requires_grad=True
    )
    valid = torch.tensor([[False, True, False, True, True, False]])
    tokens, returned = encoder.encode_with_mask(values, times, valid)
    compact_tokens = encoder(values[valid].unsqueeze(0), times[valid].unsqueeze(0))
    assert torch.equal(returned, valid)
    torch.testing.assert_close(tokens[valid], compact_tokens[0])
    assert torch.count_nonzero(tokens[~valid]) == 0
    torch.testing.assert_close(tokens.sum(dim=1) / returned.sum(dim=1, keepdim=True), compact_tokens.mean(dim=1))

    # MultiheadAttention uses key_padding_mask=~returned; SDPA uses attn_mask=returned[:, None, None, :].
    attention = torch.nn.MultiheadAttention(12, 3, batch_first=True).eval()
    query = torch.randn(1, 2, 12)
    torch.testing.assert_close(
        attention(query, tokens, tokens, key_padding_mask=~returned)[0],
        attention(query, compact_tokens, compact_tokens)[0],
    )
    q_sdpa = torch.randn(1, 3, 2, 4)
    k_pad = tokens.view(1, 6, 3, 4).transpose(1, 2)
    k_cmp = compact_tokens.view(1, 3, 3, 4).transpose(1, 2)
    torch.testing.assert_close(
        F.scaled_dot_product_attention(q_sdpa, k_pad, k_pad, attn_mask=returned[:, None, None, :]),
        F.scaled_dot_product_attention(q_sdpa, k_cmp, k_cmp),
    )

    # Use a linear phase-sensitive projection rather than orthogonal squared norms.
    probe = torch.linspace(-1.0, 1.5, 12).view(1, 1, 12)
    (tokens * probe).sum().backward()
    assert torch.count_nonzero(values.grad[~valid]) == 0
    assert torch.count_nonzero(times.grad[~valid]) == 0
    assert torch.isfinite(values.grad).all() and torch.isfinite(times.grad).all()
    assert torch.all(times.grad[:, [3, 4]].abs() > 1e-3)

    # Exact analytical derivative check on a unit pair: d/dt sin(2*pi*(t - t0)/P) = (2*pi/P)*cos(...).
    unit_enc = PhysicalTimeRoPETokenizer(output_dim=2, pinned_periods=(4.0,)).double().eval()
    with torch.no_grad():
        unit_enc.projection.weight.zero_()
        unit_enc.projection.bias.copy_(torch.tensor([1.0, 0.0], dtype=torch.float64))
    t_probe = torch.tensor([[float("nan"), 10.0, 10.5]], dtype=torch.float64, requires_grad=True)
    m_probe = torch.tensor([[False, True, True]])
    unit_tokens, _ = unit_enc.encode_with_mask(torch.ones(1, 3, dtype=torch.float64), t_probe, m_probe)
    unit_tokens[0, 2, 1].backward()
    expected_dt2 = (math.pi / 2.0) * math.cos(math.pi / 4.0)
    torch.testing.assert_close(
        t_probe.grad,
        torch.tensor([[0.0, -expected_dt2, expected_dt2]], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    # Train-mode dropout preserves exact zeros at masked positions.
    drop_enc = PhysicalTimeRoPETokenizer(input_channels=2, output_dim=12, dropout=0.5).train()
    drop_tokens, _ = drop_enc.encode_with_mask(values.detach(), times.detach(), valid)
    assert torch.count_nonzero(drop_tokens[~valid]) == 0

    none_tokens, none_valid = encoder.encode_with_mask(values.detach(), times.detach(), torch.zeros_like(valid))
    assert torch.count_nonzero(none_tokens) == 0 and not none_valid.any()


@pytest.mark.parametrize(
    "epoch, offsets, period",
    [
        (0.0, [0.0, 20 * 365.25 * 86400 + 133.125], 43200.0),
        (1.7e9, [0.0, 0.125, 17.25, 41.75], 60.0),
        (-1.1e9, [0.0, 2.2e9 + 19.375], 3600.0),
    ],
)
def test_float64_modular_oracle(epoch, offsets, period):
    """Compare token phases against a NumPy float64 modular reference over multi-decade spans."""
    encoder = PhysicalTimeRoPETokenizer(output_dim=2, pinned_periods=(period,)).eval()
    with torch.no_grad():
        encoder.projection.weight.zero_()
        encoder.projection.bias.copy_(torch.tensor([1.0, 0.0]))
    times = torch.from_numpy((epoch + np.array(offsets))[None, :])
    phase = 2 * np.pi * (np.remainder(times.numpy() - times.numpy()[:, :1], period) / period)
    expected = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
    error = np.max(np.abs(encoder(torch.ones_like(times).float(), times).detach().numpy() - expected))
    assert error < 1e-6


def test_mixed_lengths_and_empty_rows():
    """Verify mixed valid lengths, all-masked rows, and zero-length batch/sequence shapes."""
    encoder = PhysicalTimeRoPETokenizer(output_dim=8).eval()
    times = torch.tensor(
        [[float("nan"), 12.0, 24.0], [98.0, float("nan"), float("nan")], [float("nan")] * 3], dtype=torch.float64
    )
    values = times.float()
    valid = torch.isfinite(times)
    tokens, mask = encoder.encode_with_mask(values, times, valid)
    assert torch.equal(mask, valid)
    pooled = tokens.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    assert torch.isfinite(pooled).all() and not pooled[-1].any()
    for row in (0, 1):
        compact = encoder(values[row : row + 1, mask[row]], times[row : row + 1, mask[row]])
        torch.testing.assert_close(tokens[row, mask[row]], compact[0])
    empty_seq, empty_seq_mask = encoder.encode_with_mask(values[:, :0], times[:, :0])
    assert empty_seq.shape == (3, 0, 8) and empty_seq_mask.shape == (3, 0)
    empty_batch, empty_batch_mask = encoder.encode_with_mask(values[:0], times[:0])
    assert empty_batch.shape == (0, 3, 8) and empty_batch_mask.shape == (0, 3)


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (2.0, 1.0), (1.0, float("inf")), (float("nan"), 3.0)])
def test_invalid_period_bounds(bounds):
    """Reject non-positive, inverted, or non-finite period bounds."""
    with pytest.raises(ValueError, match="period bounds"):
        PhysicalTimeRoPETokenizer(min_period=bounds[0], max_period=bounds[1])


def test_constructor_and_input_validation():
    """Reject invalid dimensions, pinned periods, shapes, dtypes, and non-finite valid times."""
    with pytest.raises(ValueError, match="output_dim"):
        PhysicalTimeRoPETokenizer(output_dim=3)
    with pytest.raises(ValueError, match="input_channels"):
        PhysicalTimeRoPETokenizer(output_dim=4, input_channels=0)
    with pytest.raises(ValueError, match="pinned periods"):
        PhysicalTimeRoPETokenizer(output_dim=4, pinned_periods=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="pinned periods"):
        PhysicalTimeRoPETokenizer(output_dim=4, pinned_periods=(0.0,))

    multi = PhysicalTimeRoPETokenizer(input_channels=2, output_dim=4)
    with pytest.raises(ValueError, match=r"\[B, L, 2\]"):
        multi(torch.ones(1, 5))
    with pytest.raises(ValueError, match="mask must be boolean"):
        multi(torch.ones(1, 5, 2), mask=torch.ones(1, 5, dtype=torch.int64))
    with pytest.raises(ValueError, match="timestamps must be real-valued"):
        multi(torch.ones(1, 5, 2), timestamps=torch.ones(1, 5, dtype=torch.bool))

    tiny_period = PhysicalTimeRoPETokenizer(output_dim=2, pinned_periods=(1e-200,))
    with pytest.raises(ValueError, match="finite"):
        tiny_period(torch.ones(1, 2), torch.tensor([[float("nan"), 2.0]]))
    assert torch.isfinite(tiny_period(torch.ones(1, 2), torch.tensor([[1.0, 2.0]]))).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend is not available")
def test_mps_device_parity_and_gradients():
    """Verify MPS execution with implicit clocks, MPS float32 clocks, and CPU float64 clocks."""
    torch.manual_seed(41)
    cpu_enc = PhysicalTimeRoPETokenizer(input_channels=2, output_dim=8, pinned_periods=(60.0, 3600.0)).eval()
    mps_enc = PhysicalTimeRoPETokenizer(input_channels=2, output_dim=8, pinned_periods=(60.0, 3600.0)).to("mps").eval()
    mps_enc.load_state_dict(cpu_enc.state_dict())

    values_cpu = torch.tensor([[[float("nan"), 0.0], [1.5, -0.5], [2.0, 3.0], [0.25, -1.25]]], dtype=torch.float32)
    valid_cpu = torch.tensor([[False, True, True, True]])
    # Multi-year epoch offset with sub-minute increments requires float64 modular reduction.
    times_f64 = torch.tensor([[float("nan"), 1.7e9, 1.7e9 + 15.0, 1.7e9 + 45.0]], dtype=torch.float64)

    ref_implicit = cpu_enc(torch.ones(2, 4, 2))
    mps_implicit = mps_enc(torch.ones(2, 4, 2, device="mps"))
    assert mps_implicit.device.type == "mps"
    torch.testing.assert_close(mps_implicit.cpu(), ref_implicit, atol=1e-5, rtol=1e-5)

    values_mps = values_cpu.to("mps").detach().requires_grad_(True)
    times_cpu_grad = times_f64.clone().requires_grad_(True)
    mps_tokens, mps_valid = mps_enc.encode_with_mask(values_mps, times_cpu_grad, valid_cpu.to("mps"))
    ref_tokens, _ = cpu_enc.encode_with_mask(values_cpu, times_f64, valid_cpu)
    assert mps_tokens.device.type == "mps" and mps_valid.device.type == "mps"
    torch.testing.assert_close(mps_tokens.cpu(), ref_tokens, atol=1e-5, rtol=1e-5)

    probe = torch.linspace(-0.5, 1.25, 8, device="mps").view(1, 1, 8)
    (mps_tokens * probe).sum().backward()
    assert torch.count_nonzero(values_mps.grad[~valid_cpu.to("mps")]) == 0
    assert torch.count_nonzero(times_cpu_grad.grad[~valid_cpu]) == 0
    assert torch.isfinite(values_mps.grad).all() and torch.isfinite(times_cpu_grad.grad).all()

    # Also verify MPS-resident float32 timestamps with requires_grad=True.
    times_mps = torch.tensor([[float("nan"), 10.0, 25.0, 55.0]], device="mps", requires_grad=True)
    mps_tokens_f32, _ = mps_enc.encode_with_mask(values_cpu.to("mps"), times_mps, valid_cpu.to("mps"))
    (mps_tokens_f32 * probe).sum().backward()
    assert times_mps.grad.device.type == "mps"
    assert torch.count_nonzero(times_mps.grad[~valid_cpu.to("mps")]) == 0
    assert torch.isfinite(times_mps.grad).all()
