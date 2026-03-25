#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Test script to verify MedGemma support in OpenTSLMFlamingo.
"""

import torch
from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo


def test_medgemma_support():
    """Test that MedGemma can be loaded with OpenTSLMFlamingo."""

    # Available MedGemma models
    medgemma_models = [
        "google/medgemma-2b",
        "google/medgemma-7b",
        "google/medgemma-27b",
    ]

    print("🧪 Testing MedGemma support in OpenTSLMFlamingo")
    print("=" * 60)

    for model_id in medgemma_models:
        try:
            print(f"\n🔍 Testing {model_id}...")

            # Try to initialize the model
            model = OpenTSLMFlamingo(
                device="cpu",  # Use CPU for testing
                llm_id=model_id,
                cross_attn_every_n_layers=1,
            )

            print(f"✅ Successfully loaded {model_id}")
            print(f"   Model type: {type(model.llm).__name__}")
            print(f"   Tokenizer vocab size: {len(model.text_tokenizer)}")

            # Test basic functionality
            test_batch = [
                {
                    "pre_prompt": "You are an expert in time series analysis.",
                    "time_series_text": [
                        "This is a test time series with mean 0.0 and std 1.0:"
                    ],
                    "post_prompt": "Please analyze this time series.",
                    "answer": "This appears to be a normalized time series.",
                    "time_series": [torch.randn(100)],  # Random test data
                }
            ]

            # Test forward pass
            with torch.no_grad():
                loss = model.compute_loss(test_batch)
                print(f"   Test loss: {loss.item():.4f}")

            # Test generation
            with torch.no_grad():
                predictions = model.generate(test_batch, max_new_tokens=10)
                print(f"   Test generation: {predictions[0][:50]}...")

            print(f"✅ All tests passed for {model_id}")

        except Exception as e:
            print(f"❌ Failed to load {model_id}: {e}")
            continue

    print("\n🎉 MedGemma support test completed!")


if __name__ == "__main__":
    test_medgemma_support()
