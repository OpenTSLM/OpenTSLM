#!/usr/bin/env python3
"""Fine-tune Gemma on HAR (Human Activity Recognition) dataset with LoRA."""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

try:
    from common_finetune_sft import run_sft
except ModuleNotFoundError:
    from evaluation.baseline.common_finetune_sft import run_sft


def _time_series_to_pil(time_series) -> Image.Image:
    """Render a 1D or 2D time series array to a PIL RGB image."""
    if isinstance(time_series, np.ndarray):
        if time_series.ndim == 1:
            ts_list = [time_series]
        elif time_series.ndim == 2:
            ts_list = [time_series[i] for i in range(time_series.shape[0])]
        else:
            raise ValueError(f"Unsupported ndarray shape: {time_series.shape}")
    else:
        ts_list = list(time_series)
        if len(ts_list) > 0 and not hasattr(ts_list[0], "__len__"):
            ts_list = [ts_list]

    n = len(ts_list)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), dpi=100, sharex=True)
    if n == 1:
        axes = [axes]
    for i, s in enumerate(ts_list):
        s = np.asarray(s)
        axes[i].plot(s, marker="o", linestyle="-", markersize=0)
        axes[i].axis("off")
    plt.tight_layout(pad=0.1)

    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


def _build_messages_from_sample(sample: dict, eos_token: str = "") -> dict:
    """Build chat-style messages with accelerometer plot for training."""
    pre = (sample.get("pre_prompt") or "").strip()
    post = (sample.get("post_prompt") or "").strip()
    ans = (sample.get("answer") or "").strip()

    if eos_token:
        ans = ans + eos_token

    x_axis = sample.get("x_axis", [])
    y_axis = sample.get("y_axis", [])
    z_axis = sample.get("z_axis", [])

    ts = np.array([x_axis, y_axis, z_axis])
    img = _time_series_to_pil(ts)

    user_text = "\n\n".join([p for p in [pre, post] if p])
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful AI that analyzes accelerometer data for human activity recognition."}],
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
    parser = argparse.ArgumentParser(description="Fine-tune Gemma on HAR dataset with LoRA")
    parser.add_argument("--output-dir", type=str, default="runs/gemma3-4b-pt-har-lora")
    parser.add_argument("--llm-id", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    eos_token = processor.tokenizer.eos_token

    ds = HARCoTQADataset(split="train", EOS_TOKEN="")
    n = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
    train_examples = [_build_messages_from_sample(ds[i], eos_token=eos_token) for i in range(n)]

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
