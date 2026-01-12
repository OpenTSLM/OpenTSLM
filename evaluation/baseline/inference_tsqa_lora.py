#!/usr/bin/env python3
"""
Inference script for the fine-tuned LoRA model on TSQA data.
Loads the base model + LoRA adapters and runs inference on new examples.
"""
import os
import sys
import json
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from PIL import Image
import numpy as np
import csv
from tqdm import tqdm

# Add project paths
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from time_series_datasets.TSQADataset import TSQADataset

# Import the helper function from finetune_tsqa_plot
try:
    from finetune_tsqa_plot import _time_series_to_pil
except ModuleNotFoundError:
    from evaluation.baseline.finetune_tsqa_plot import _time_series_to_pil


def load_model_and_processor(base_model_id: str, lora_adapter_path: str = None):
    """Load the base model with optional LoRA adapters and processor.
    
    Args:
        base_model_id: HuggingFace model ID for the base model
        lora_adapter_path: Path to LoRA adapters. If None, loads only the base model.
    """
    print(f"Loading base model: {base_model_id}")
    
    # Load base model
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA adapters if path is provided
    if lora_adapter_path:
        print(f"Loading LoRA adapters from: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        # Load processor from LoRA path
        processor = AutoProcessor.from_pretrained(lora_adapter_path)
    else:
        print("No LoRA adapters specified - using base model only")
        # Load processor from base model
        processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    
    model.eval()  # Set to evaluation mode
    
    print("Model and processor loaded successfully!")
    return model, processor


def run_inference(model, processor, messages, max_new_tokens=512, temperature=0.7):
    """
    Run inference on a single example.
    
    Args:
        model: The loaded model with LoRA adapters
        processor: The processor for tokenization and image processing
        messages: List of message dicts in chat format (can include images)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text response
    """
    # Extract images from messages
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                image = element.get("image", element)
                if image is not None and hasattr(image, "convert"):
                    images.append(image.convert("RGB"))
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    
    # Process inputs
    inputs = processor(
        text=text,
        images=images if images else None,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (remove the prompt)
    prompt_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
    if generated_text.startswith(prompt_text):
        generated_text = generated_text[len(prompt_text):].strip()
    
    return generated_text


def _build_messages_from_sample_for_inference(sample: dict) -> tuple:
    """Build chat-style messages using pre/post prompts and an image of the time series.
    Returns (messages, user_text) for logging.
    """
    pre = (sample.get("pre_prompt") or "").strip()
    post = (sample.get("post_prompt") or "").strip()

    # Get time series
    time_series = sample.get("time_series")
    if time_series is None and "Series" in sample:
        try:
            time_series = json.loads(sample["Series"])  # raw TSQA field
        except Exception:
            time_series = []

    # Fallbacks for other potential keys used in codebase
    if time_series is None:
        time_series = sample.get("original_data", sample.get("signal", []))

    img = _time_series_to_pil(time_series)

    user_text = "\n\n".join([pre, post])

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI that analyzes time series data to answer questions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": img},
            ],
        },
    ]
    return messages, user_text


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run inference with LoRA model on TSQA data")
    parser.add_argument("--base-model", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--lora-path", type=str, default="runs/gemma3-4b-pt-tsqa-lora",
                        help="Path to LoRA adapters. Set to 'none' or empty to use base model only.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "validation"])
    parser.add_argument("--output-csv", type=str, default="inference_tsqa_results.csv",
                        help="Path to save results CSV file")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    args = parser.parse_args()
    
    # Configuration
    BASE_MODEL_ID = args.base_model
    # Handle 'none' or empty string as no LoRA
    LORA_ADAPTER_PATH = args.lora_path if args.lora_path and args.lora_path.lower() != 'none' else None
    
    # Load model and processor
    model, processor = load_model_and_processor(BASE_MODEL_ID, LORA_ADAPTER_PATH)
    
    # Load TSQA dataset
    print(f"\nLoading TSQA {args.split} split...")
    ds = TSQADataset(split=args.split, EOS_TOKEN="")
    print(f"Dataset size: {len(ds)}")
    
    # Determine number of samples to process
    num_samples = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
    print(f"Processing {num_samples} samples...\n")
    
    # Store results
    results = []
    
    # Process all samples
    for idx in tqdm(range(num_samples), desc="Running inference"):
        sample = ds[idx]
        
        # Extract prompts and data from the sample
        pre_prompt = (sample.get("pre_prompt") or "").strip()
        post_prompt = (sample.get("post_prompt") or "").strip()
        ground_truth = (sample.get("answer") or sample.get("label") or "").strip()
        
        # Build messages and user text
        messages, user_text = _build_messages_from_sample_for_inference(sample)
        
        # Run inference
        response = run_inference(model, processor, messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        
        # Store result
        result = {
            "sample_idx": idx,
            "input_text": user_text,
            "target_answer": ground_truth,
            "generated_answer": response,
        }
        results.append(result)
        
        # Print first 5 results
        if idx < 5:
            print("\n" + "="*80)
            print(f"SAMPLE {idx} from {args.split} split")
            print("="*80)
            print(f"\nQUESTION:\n{user_text}")
            print(f"\nGROUND TRUTH ANSWER:\n{ground_truth}")
            print(f"\nMODEL RESPONSE:\n{response}")
            print("="*80)
    
    # Save results to CSV
    print(f"\n\nSaving results to {args.output_csv}...")
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["sample_idx", "input_text", "target_answer", "generated_answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"✓ Saved {len(results)} results to {args.output_csv}")
    print("\n" + "="*80)
    print("TSQA inference completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
