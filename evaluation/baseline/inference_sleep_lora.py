#!/usr/bin/env python3
"""Inference script for fine-tuned LoRA model on SleepEDF data."""

import os
import sys
import argparse
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import csv
from tqdm import tqdm

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset

try:
    from finetune_sleep_plot import _time_series_to_pil
except ModuleNotFoundError:
    from evaluation.baseline.finetune_sleep_plot import _time_series_to_pil


def load_model_and_processor(base_model_id: str, lora_adapter_path: str = None):
    """Load base model with optional LoRA adapters."""
    print(f"Loading base model: {base_model_id}")

    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if lora_adapter_path:
        print(f"Loading LoRA adapters from: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        processor = AutoProcessor.from_pretrained(lora_adapter_path)
    else:
        print("No LoRA adapters specified - using base model only")
        processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    model.eval()
    print("Model and processor loaded successfully!")
    return model, processor


def run_inference(model, processor, messages, max_new_tokens=512, temperature=0.7):
    """Run inference on a single example and return generated text."""
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

    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    inputs = processor(
        text=text,
        images=images if images else None,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    prompt_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
    if generated_text.startswith(prompt_text):
        generated_text = generated_text[len(prompt_text):].strip()

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Run inference with LoRA model on SleepEDF data")
    parser.add_argument("--base-model", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--lora-path", type=str, default="runs/gemma3-4b-pt-sleep-lora",
                        help="Path to LoRA adapters. Set to 'none' or empty to use base model only.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "validation"])
    parser.add_argument("--output-csv", type=str, default="inference_results.csv",
                        help="Path to save results CSV file")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    args = parser.parse_args()

    base_model_id = args.base_model
    lora_adapter_path = args.lora_path if args.lora_path and args.lora_path.lower() != 'none' else None

    model, processor = load_model_and_processor(base_model_id, lora_adapter_path)

    print(f"\nLoading SleepEDF {args.split} split...")
    ds = SleepEDFCoTQADataset(split=args.split, EOS_TOKEN="")
    print(f"Dataset size: {len(ds)}")

    num_samples = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
    print(f"Processing {num_samples} samples...\n")

    results = []

    for idx in tqdm(range(num_samples), desc="Running inference"):
        sample = ds[idx]

        pre_prompt = (sample.get("pre_prompt") or "").strip()
        post_prompt = (sample.get("post_prompt") or "").strip()
        ground_truth = sample.get("label", "Unknown")

        ts = sample.get("original_data", sample.get("time_series", None))
        sleep_image = _time_series_to_pil(ts)

        user_text = "\n\n".join([pre_prompt, post_prompt])

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful medical AI that analyzes sleep EEG."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": sleep_image},
                ]
            }
        ]

        response = run_inference(model, processor, messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

        result = {
            "sample_idx": idx,
            "input_text": user_text,
            "target_answer": ground_truth,
            "generated_answer": response,
        }
        results.append(result)

        if idx < 5:
            print("\n" + "="*80)
            print(f"SAMPLE {idx} from {args.split} split")
            print("="*80)
            print(f"\nGround Truth Label: {ground_truth}")
            print(f"\nQUESTION:\n{user_text}")
            print(f"\nGT REASONING:\n{sample.get('answer', '')}")
            print(f"\nMODEL RESPONSE:\n{response}")
            print("="*80)

    print(f"\n\nSaving results to {args.output_csv}...")
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["sample_idx", "input_text", "target_answer", "generated_answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Saved {len(results)} results to {args.output_csv}")
    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
