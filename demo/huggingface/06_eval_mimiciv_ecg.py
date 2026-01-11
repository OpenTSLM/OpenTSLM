# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""
Evaluation script for testing OpenTSLM models on MimicIV-ECG dataset with custom prompts.

This script:
1. Loads a pretrained model from HuggingFace Hub or local checkpoint
2. Loads the MimicIV-ECG dataset
3. Runs inference with custom prompts on all samples
4. Stores outputs for later evaluation
"""

import sys
import os
from pathlib import Path

# Add src directory to path to include new mimiciv_ecg module
# This allows the script to work even when opentslm is installed via pip
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
src_dir = project_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

import argparse
import json
from typing import List, Optional
import torch
from tqdm import tqdm

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.time_series_datasets.mimiciv_ecg.MimicIVECGDataset import MimicIVECGDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from torch.utils.data import DataLoader
from opentslm.model_config import PATCH_SIZE


# Define your custom prompts here
# These will be applied to each sample in the dataset
CUSTOM_PROMPTS = [
    "Summarize this ECG and highlight any clinically relevant findings.",
    "Provide an overall interpretation of this ECG.",
    "What are your findings on this ECG?",
    "Describe this ECG in a clinically meaningful way.",
    "Interpret this ECG and describe any abnormalities.",
    "What stands out as abnormal on this ECG?",
    "What are the key findings on this ECG?",
    "Are there any clinically significant abnormalities on this ECG?",
    "Based on this ECG, what is the most likely diagnosis?",
    "What diagnoses should be considered based on this ECG?",
    "Does this ECG show signs of arrhythmia?",
    "Does this ECG show signs of atrial fibrillation?",
    "Are there any ST-segment or T-wave abnormalities?",
    "Are there any signs of ischemia or myocardial infarction?",
    "Does this ECG show evidence of structural heart disease?",
]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OpenTSLM model on MimicIV-ECG dataset with custom prompts"
    )
    parser.add_argument(
        "--model_repo",
        type=str,
        default="OpenTSLM/llama-3.2-1b-ecg-sp",
        help="HuggingFace repository ID or local path to model",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "validation"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--ecg_data_dir",
        type=str,
        default=None,
        help="Directory containing MimicIV-ECG data files (.dat/.hea). "
             "If not provided, will try default location.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to use for all samples (overrides CUSTOM_PROMPTS list)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/mimiciv_eval",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not provided.",
    )
    parser.add_argument(
        "--no_fix_lead_order",
        action="store_true",
        help="Don't reorder leads to match ECG-QA order",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MimicIV-ECG Evaluation Script")
    print("=" * 80)
    
    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"\n📥 Loading model from {args.model_repo}...")
    enable_lora = False
    if "-sp" in args.model_repo or "sp" in args.model_repo.lower():
        enable_lora = True
    
    try:
        model = OpenTSLM.load_pretrained(
            args.model_repo,
            enable_lora=enable_lora,
            device=device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model repository ID is correct or provide a local path.")
        return
    
    # Determine which prompts to use
    if args.prompt:
        # Single prompt for all samples
        custom_prompts = [args.prompt]
        print(f"\n📝 Using single custom prompt: {args.prompt[:100]}...")
    else:
        # Use prompts defined in CUSTOM_PROMPTS list
        custom_prompts = CUSTOM_PROMPTS
        print(f"\n📝 Using {len(custom_prompts)} custom prompts from CUSTOM_PROMPTS list")
    
    # Create dataset
    print(f"\n📊 Loading MimicIV-ECG {args.split} dataset...")
    try:
        test_dataset = MimicIVECGDataset(
            split=args.split,
            EOS_TOKEN=model.get_eos_token(),
            max_samples=args.max_samples,
            ecg_data_dir=args.ecg_data_dir,
            custom_prompts=custom_prompts,
            fix_lead_order=not args.no_fix_lead_order,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure MimicIV-ECG data is available in data/mimiciv_ecg/")
        print("2. Set --ecg_data_dir to point to directory containing .dat/.hea files")
        print("3. Check that ECG files exist for the ECG IDs in the dataset")
        return
    
    print(f"Loaded {len(test_dataset)} samples")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare output file
    output_file = os.path.join(
        args.output_dir,
        f"mimiciv_eval_{args.split}_{args.model_repo.split('/')[-1]}.jsonl"
    )
    
    print(f"\n🔍 Running inference on {len(test_dataset)} samples...")
    print(f"💾 Results will be saved to: {output_file}")
    print("=" * 80)
    
    all_results = []
    
    # Iterate over dataset
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        try:
            # Generate predictions
            predictions = model.generate(batch, max_new_tokens=args.max_new_tokens)
            
            # Debug: Print first batch predictions
            if batch_idx == 0:
                print(f"\n🔍 DEBUG: First batch predictions:")
                print(f"  Number of predictions: {len(predictions)}")
                for i, pred in enumerate(predictions):
                    print(f"  Prediction {i}: {repr(pred[:200]) if pred else 'EMPTY'}")
                    print(f"  Prediction {i} length: {len(pred) if pred else 0}")
            
            # Process each sample in the batch
            for sample_idx, (sample, pred) in enumerate(zip(batch, predictions)):
                result = {
                    "sample_id": sample.get("sample_id", batch_idx * args.batch_size + sample_idx),
                    "ecg_id": sample.get("ecg_id", "N/A"),
                    "template_id": sample.get("template_id", "N/A"),
                    "question_type": sample.get("question_type", "N/A"),
                    "pre_prompt": sample.get("pre_prompt", ""),
                    "post_prompt": sample.get("post_prompt", ""),
                    "model_output": pred,
                    "gold_answer": sample.get("answer", ""),
                }
                
                all_results.append(result)
                
                # Write incrementally to file
                with open(output_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
        
        except Exception as e:
            print(f"\n⚠️  Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Also save as JSON for easier loading
    json_output_file = output_file.replace('.jsonl', '.json')
    with open(json_output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Processed {len(all_results)} samples")
    print(f"   Results saved to:")
    print(f"     - {output_file} (JSONL format)")
    print(f"     - {json_output_file} (JSON format)")
    
    # Print summary
    if len(all_results) > 0:
        print(f"\n📊 Summary:")
        print(f"   Total samples: {len(all_results)}")
        print(f"   Model: {args.model_repo}")
        print(f"   Split: {args.split}")
        if custom_prompts:
            print(f"   Custom prompts: {len(custom_prompts)}")
        print(f"\n   Sample outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
