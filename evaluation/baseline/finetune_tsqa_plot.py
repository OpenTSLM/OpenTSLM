#!/usr/bin/env python3
#
# Fine-tune Gemma on the TSQA dataset with LoRA
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

"""Also add project root so we can import sibling modules when running script directly."""
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from time_series_datasets.TSQADataset import TSQADataset

# Prefer local import when running from evaluation/baseline, fall back to package path
try:
    from common_finetune_sft import run_sft  # when cwd is this folder or script path is used
except ModuleNotFoundError:
    from evaluation.baseline.common_finetune_sft import run_sft


def _time_series_to_pil(time_series) -> Image.Image:
    """Render a 1D time series array to a PIL RGB image (no disk I/O)."""
    # Ensure time_series is a list of numbers
    if isinstance(time_series, np.ndarray):
        if time_series.ndim > 1:
            time_series = time_series[0]  # Take first channel if multi-channel
    else:
        time_series = np.array(time_series)
        if len(time_series.shape) > 1:
            time_series = time_series[0]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=100)
    ax.plot(time_series, marker="o", linestyle="-", markersize=0)
    ax.axis("off")
    plt.tight_layout(pad=0.1)

    # Convert to PIL Image
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img

def _build_messages_from_sample(sample: dict, eos_token: str = "") -> dict:
    """Build chat-style messages using pre/post prompts and an image of the time series."""
    pre = (sample.get("pre_prompt") or "").strip()
    post = (sample.get("post_prompt") or "").strip()
    ans = (sample.get("answer") or "").strip()
    
    # Append EOS token to answer if provided
    if eos_token:
        ans = ans + eos_token

    # Get the time series data
    time_series = sample.get("time_series", [])
    if not time_series and "Series" in sample:
        # Handle case where raw Series data is available
        import json
        time_series = json.loads(sample["Series"])
    
    # Convert time series to image
    img = _time_series_to_pil(time_series)

    # Build messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful AI that analyzes time series data to answer questions."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{pre}\n\n{post}"},
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
        description="Fine-tune Gemma on the TSQA dataset with LoRA"
    )
    # SFT options
    parser.add_argument("--output-dir", type=str, default="runs/gemma3-4b-pt-tsqa-lora")
    parser.add_argument("--llm-id", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--max-samples", type=int, default=1000)
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

    # Build training chat examples with images from TSQA train split
    ds = TSQADataset(split="train", EOS_TOKEN="")
    n = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
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
