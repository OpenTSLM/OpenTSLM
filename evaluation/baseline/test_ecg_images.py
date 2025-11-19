#!/usr/bin/env python3
"""Test script to verify ECG-QA image pipeline.

Saves images at two stages:
1. Original: directly from _ecg_leads_to_pil
2. Extracted: after going through messages dict and process_vision_info

Run: python evaluation/baseline/test_ecg_images.py
Check: test_images/ folder - images should match
"""

import os
import sys

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset

from PIL import Image

try:
    from finetune_ecg_qa_plot import _get_ecg_id_from_sample, _build_messages_from_sample
    from common_finetune_sft import process_vision_info, _load_ecg_data, _ecg_to_image
except ModuleNotFoundError:
    from evaluation.baseline.finetune_ecg_qa_plot import _get_ecg_id_from_sample, _build_messages_from_sample
    from evaluation.baseline.common_finetune_sft import process_vision_info, _load_ecg_data, _ecg_to_image


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test ECG-QA image pipeline")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to test")
    parser.add_argument("--output-dir", type=str, default="test_images", help="Output directory for images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ECG-QA dataset...")
    ds = ECGQACoTQADataset(split="train", EOS_TOKEN="", max_samples=args.num_samples)
    print(f"Loaded {len(ds)} samples\n")

    for i in range(min(args.num_samples, len(ds))):
        sample = ds[i]
        print(f"Sample {i}:")

        # Stage 1: Generate image directly from raw ECG data
        ecg_id = _get_ecg_id_from_sample(sample)
        ecg_data = _load_ecg_data(ecg_id)
        original_img = _ecg_to_image(ecg_data)
        original_path = os.path.join(args.output_dir, f"sample_{i}_original.png")
        original_img.save(original_path)
        print(f"  Original image: {original_path} ({original_img.size})")

        # Stage 2: Build messages and extract via process_vision_info
        messages_dict = _build_messages_from_sample(sample, eos_token="")
        messages = messages_dict["messages"]
        extracted_images = process_vision_info(messages)

        if extracted_images:
            extracted_img = extracted_images[0]
            extracted_path = os.path.join(args.output_dir, f"sample_{i}_extracted.png")
            extracted_img.save(extracted_path)
            print(f"  Extracted image: {extracted_path} ({extracted_img.size})")

            # Quick check if they're the same size
            if original_img.size == extracted_img.size:
                print(f"  ✓ Sizes match")
            else:
                print(f"  ✗ Size mismatch!")
        else:
            print(f"  ✗ No image extracted from messages!")

        print()

    print(f"Done! Check images in: {args.output_dir}/")


if __name__ == "__main__":
    main()
