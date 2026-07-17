#!/usr/bin/env python3
#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Unit tests for the Chronos2Encoder implementation.

These tests validate:
- Basic initialization on CPU
- Forward pass shape and dtype
- Freeze vs. non-freeze backbone behavior
"""

import os
import sys
import torch

# Make sure we can import the project as a package.
# We add both the project root (so `src.*` works) and `src/` itself (so `model.*` works),
# matching how main scripts manipulate sys.path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.model.encoder.Chronos2Encoder import Chronos2Encoder, CHRONOS_AVAILABLE  # noqa: E402
from src.model_config import ENCODER_OUTPUT_DIM  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _skip_if_no_chronos() -> bool:
    if not CHRONOS_AVAILABLE:
        print("⚠️  chronos-forecasting is not installed, skipping Chronos2Encoder tests")
        return True
    return False


def test_chronos2_encoder_init_cpu():
    """Test that Chronos2Encoder can be initialized on CPU."""
    print("🧪 Testing Chronos2Encoder CPU initialization...")
    if _skip_if_no_chronos():
        return True

    try:
        encoder = Chronos2Encoder(device="cpu")
        assert isinstance(encoder, Chronos2Encoder)

        # Check that projection layer has correct output dim
        assert encoder.projection.out_features == ENCODER_OUTPUT_DIM

        print("✅ Chronos2Encoder CPU initialization successful")
        return True
    except Exception as e:
        print(f"❌ Chronos2Encoder CPU initialization failed: {e}")
        return False


def test_chronos2_encoder_forward_shape():
    """Test that forward pass returns expected shape [B, N, ENCODER_OUTPUT_DIM]."""
    print("🧪 Testing Chronos2Encoder forward shape...")
    if _skip_if_no_chronos():
        return True

    try:
        encoder = Chronos2Encoder(device=DEVICE)
        encoder.eval()

        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, device=DEVICE)

        with torch.no_grad():
            out = encoder(x)

        assert out.ndim == 3, f"Expected 3D tensor, got shape {tuple(out.shape)}"
        B, N, D = out.shape
        assert B == batch_size, f"Batch dim mismatch: expected {batch_size}, got {B}"
        assert D == ENCODER_OUTPUT_DIM, f"Embedding dim mismatch: expected {ENCODER_OUTPUT_DIM}, got {D}"
        assert N > 0, "Number of patches/tokens N should be > 0"

        print(f"✅ Forward shape OK: {tuple(out.shape)}")
        return True
    except Exception as e:
        print(f"❌ Chronos2Encoder forward shape test failed: {e}")
        return False


def test_chronos2_encoder_freeze_backbone():
    """Test that freeze_backbone correctly toggles requires_grad on Chronos-2 backbone."""
    print("🧪 Testing Chronos2Encoder freeze_backbone behavior...")
    if _skip_if_no_chronos():
        return True

    try:
        # Case 1: freeze_backbone=True
        enc_frozen = Chronos2Encoder(device=DEVICE, freeze_backbone=True)
        backbone_params = list(enc_frozen.chronos_model.parameters())
        assert backbone_params, "Chronos-2 model has no parameters?"

        # All backbone params should be frozen
        assert all(not p.requires_grad for p in backbone_params), "Backbone parameters should be frozen"
        # Projection should still be trainable
        assert enc_frozen.projection.weight.requires_grad, "Projection layer should remain trainable"

        # Case 2: freeze_backbone=False
        enc_trainable = Chronos2Encoder(device=DEVICE, freeze_backbone=False)
        backbone_params2 = list(enc_trainable.chronos_model.parameters())
        assert any(p.requires_grad for p in backbone_params2), "Backbone parameters should be trainable when not frozen"

        print("✅ freeze_backbone behavior is correct")
        return True
    except Exception as e:
        print(f"❌ Chronos2Encoder freeze_backbone test failed: {e}")
        return False


def main():
    print("🚀 Running Chronos2Encoder tests")
    print("=" * 50)

    tests = [
        test_chronos2_encoder_init_cpu,
        test_chronos2_encoder_forward_shape,
        test_chronos2_encoder_freeze_backbone,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print(f"\n📊 Chronos2Encoder Test Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All Chronos2Encoder tests passed!")
    else:
        print("⚠️  Some Chronos2Encoder tests failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


