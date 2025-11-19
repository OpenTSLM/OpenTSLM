#!/usr/bin/env python3
#
# Fine-tune Gemma on the ECG-QA CoT dataset with LoRA
#

import os
import sys
import argparse

# Ensure project src/ is on sys.path so we can import time_series_datasets
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# Also add project root so we can import sibling modules when running script directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset

# Prefer local import when running from evaluation/baseline, fall back to package path
try:
    from common_finetune_sft import run_sft  # when cwd is this folder or script path is used
except ModuleNotFoundError:
    from evaluation.baseline.common_finetune_sft import run_sft


def _get_ecg_id_from_sample(sample: dict) -> int:
    """Extract ECG ID from a sample dict."""
    ecg_id = sample.get("ecg_id")
    if ecg_id is None:
        raise ValueError("Sample missing 'ecg_id' field")

    if isinstance(ecg_id, list):
        if len(ecg_id) == 0:
            raise ValueError("Sample 'ecg_id' list is empty")
        return ecg_id[0]

    return ecg_id


def _build_messages_from_sample(sample: dict, eos_token: str = "") -> dict:
    """Build chat-style messages with lazy ECG ID reference (no image rendering upfront)."""
    pre = (sample.get("pre_prompt") or "").strip()
    post = (sample.get("post_prompt") or "").strip()
    ans = (sample.get("answer") or "").strip()

    if eos_token:
        ans = ans + eos_token

    # Store only the ECG ID - image will be rendered lazily in collate_fn
    ecg_id = _get_ecg_id_from_sample(sample)

    question = sample.get("question")
    if question:
        pre_text = f"{pre}\n\nQuestion: {question}" if pre else f"Question: {question}"
    else:
        pre_text = pre

    user_text = "\n\n".join([p for p in [pre_text, post] if p])
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful medical AI that analyzes ECG time series to answer cardiology questions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "ecg_id": ecg_id},  # Lazy reference - no bytes yet
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ans}],
        },
    ]
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma on the ECG-QA CoT dataset with LoRA"
    )
    # SFT options
    parser.add_argument("--output-dir", type=str, default="runs/gemma3-4b-pt-ecgqa-lora")
    parser.add_argument("--llm-id", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--max-samples", type=int, default=1000, help="Number of samples to process in this run")
    parser.add_argument("--exclude-comparison", action="store_true", help="Exclude comparison-type ECG-QA questions")
    parser.add_argument("--preload-processed-data", action="store_true", help="Preload processed ECG data (faster, more RAM). Default off.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=4096)

    args = parser.parse_args()

    # Load processor to get EOS token
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    eos_token = processor.tokenizer.eos_token

    # Load dataset
    ds = ECGQACoTQADataset(
        split="train",
        EOS_TOKEN="",
        max_samples=args.max_samples,
        exclude_comparison=args.exclude_comparison,
        preload_processed_data=args.preload_processed_data,
    )

    print(f"Processing {len(ds)} samples (lazy image loading - images rendered on-demand)")

    train_examples = [_build_messages_from_sample(ds[i], eos_token=eos_token) for i in range(len(ds))]

    # Run SFT training
    run_sft(
        train_examples,
        output_dir=args.output_dir,
        llm_id=args.llm_id,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_len=args.max_seq_len,
        save_steps=500,
    )


if __name__ == "__main__":
    main()
