#!/usr/bin/env python3
#
# Fine-tune Gemma on the ECG-QA CoT dataset with LoRA
#

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io

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


def _ecg_leads_to_pil(leads: list[np.ndarray]) -> Image.Image:
    """Render multiple ECG leads to a single PIL RGB image."""
    ts_list = []
    for s in leads:
        try:
            arr = np.asarray(s, dtype=float).reshape(-1)
            if arr.size == 0 or not np.isfinite(arr).all():
                continue
            ts_list.append(arr)
        except Exception:
            continue

    if len(ts_list) == 0:
        # Fallback to a blank image to avoid crashes
        fig, ax = plt.subplots(figsize=(10, 2), dpi=100)
        ax.text(0.5, 0.5, "No ECG data", ha="center", va="center")
        ax.axis("off")
        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    n = len(ts_list)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n), dpi=100, sharex=True)
    if n == 1:
        axes = [axes]
    for i, s in enumerate(ts_list):
        axes[i].plot(s, color="black", linewidth=1.0)
        axes[i].axis("off")
    plt.tight_layout(pad=0.2)

    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


def _extract_ecg_series_from_sample(sample: dict, max_leads: int = 12) -> list:
    """Extract 1D ECG lead arrays from a sample dict."""
    series_list = []

    ts_items = sample.get("time_series_text") or []
    for item in ts_items:
        arr = None
        if isinstance(item, dict):
            arr = item.get("time_series")
        else:
            arr = getattr(item, "time_series", None)
        if arr is not None:
            try:
                a = np.asarray(arr, dtype=float).reshape(-1)
                if a.size > 0 and np.isfinite(a).all():
                    series_list.append(a)
            except Exception:
                continue
        if len(series_list) >= max_leads:
            break

    if not series_list:
        for key in ("time_series", "original_data", "signal"):
            if key in sample and sample[key] is not None:
                maybe = sample[key]
                if isinstance(maybe, (list, tuple, np.ndarray)):
                    arr = np.asarray(maybe)
                    if arr.ndim == 2:
                        for i in range(min(arr.shape[0], max_leads)):
                            series_list.append(arr[i].reshape(-1))
                    elif arr.ndim == 1:
                        series_list.append(arr.reshape(-1))
                break

    return series_list[:max_leads]


def _build_messages_from_sample(sample: dict, eos_token: str = "") -> dict:
    """Build chat-style messages with ECG plot image for training."""
    pre = (sample.get("pre_prompt") or "").strip()
    post = (sample.get("post_prompt") or "").strip()
    ans = (sample.get("answer") or "").strip()

    if eos_token:
        ans = ans + eos_token

    leads = _extract_ecg_series_from_sample(sample, max_leads=12)
    img = _ecg_leads_to_pil(leads)

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
                {"type": "image", "image": img},
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
    parser.add_argument("--max-samples", type=int, default=1000)
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

    ds = ECGQACoTQADataset(
        split="train",
        EOS_TOKEN="",
        max_samples=args.max_samples,
        exclude_comparison=args.exclude_comparison,
        preload_processed_data=args.preload_processed_data,
    )
    n = len(ds)
    train_examples = [_build_messages_from_sample(ds[i], eos_token=eos_token) for i in range(n)]

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
    )


if __name__ == "__main__":
    main()
