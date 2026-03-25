# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""
Demo script for testing HAR CoT (Human Activity Recognition Chain-of-Thought) model from HuggingFace.

This script:
1. Loads a pretrained model from HuggingFace Hub
2. Loads the HAR CoT test dataset
3. Generates predictions on the evaluation set
4. Prints model outputs
"""

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from torch.utils.data import DataLoader
from opentslm.model_config import PATCH_SIZE
import torch

# Model repository ID - change this to test different models
REPO_ID = "OpenTSLM/llama-3.2-1b-har-sp"

def main():
    print("=" * 60)
    print("HAR CoT Model Demo")
    print("=" * 60)
    
    # Load model from HuggingFace
    print(f"\n📥 Loading model from {REPO_ID}...")
    enable_lora = False
    if "-sp" in REPO_ID:
        enable_lora = True
    model = OpenTSLM.load_pretrained(REPO_ID, enable_lora=enable_lora, device="cuda" if torch.cuda.is_available() else "cpu")

   
    
    # Create dataset
    print("\n📊 Loading HAR CoT test dataset...")
    test_dataset = HARCoTQADataset("test", EOS_TOKEN=model.get_eos_token())
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )
    
    print(f"\n🔍 Running inference on {len(test_dataset)} test samples...")
    print("=" * 60)
    
    # Iterate over evaluation set
    for i, batch in enumerate(test_loader):
        # Generate predictions
        predictions = model.generate(batch, max_new_tokens=500)
        
        # Print results
        for sample, pred in zip(batch, predictions):
            print(f"\n📝 Sample {i + 1}:")
            if 'pre_prompt' in sample:
                print(f"   Question: {sample['pre_prompt']}")
            print(f"   Gold Answer: {sample.get('answer', 'N/A')}")
            print(f"   Model Output: {pred}")
            print("-" * 60)
        
        # Limit to first 5 samples for demo
        if i >= 4:
            print("\n✅ Demo complete! (Showing first 5 samples)")
            break

if __name__ == "__main__":
    main()

