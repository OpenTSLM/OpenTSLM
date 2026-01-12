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
    """Render multiple ECG leads (1D arrays) to a single PIL RGB image without disk I/O.

    - Accepts a list of 1D arrays. Non-numeric or empty arrays are skipped.
    - Arranges subplots in a vertical stack (up to 12 leads typical), sharing x-axis.
    """
    # Filter/normalize input
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
    """Extract a list of 1D ECG lead arrays from the formatted sample.

    The ECGQACoTQADataset formats samples via PromptWithAnswer.to_dict(). It typically
    provides a list under the key `time_series_text` where each item pairs text with a
    1D array. Items may be objects or dict-like; this function handles both.
    """
    series_list = []
    print("[DBG] Extracting ECG series from sample ...")

    # Preferred structured field
    ts_items = sample.get("time_series_text") or []
    print(f"[DBG] time_series_text items: {len(ts_items)}")
    for item in ts_items:
        arr = None
        # dict-like
        if isinstance(item, dict):
            arr = item.get("time_series")
        else:
            # object-like with attribute
            arr = getattr(item, "time_series", None)
        if arr is not None:
            try:
                a = np.asarray(arr, dtype=float).reshape(-1)
                if a.size > 0 and np.isfinite(a).all():
                    series_list.append(a)
                    print(f"[DBG]  + lead len={a.size}")
            except Exception:
                continue
        if len(series_list) >= max_leads:
            break

    # Fallbacks: if nothing found, try generic fields used elsewhere
    if not series_list:
        print("[DBG] No leads found in time_series_text; trying generic fields ...")
        for key in ("time_series", "original_data", "signal"):
            if key in sample and sample[key] is not None:
                maybe = sample[key]
                if isinstance(maybe, (list, tuple, np.ndarray)):
                    arr = np.asarray(maybe)
                    # If 2D (leads x time), split rows; if 1D, wrap single
                    if arr.ndim == 2:
                        for i in range(min(arr.shape[0], max_leads)):
                            series_list.append(arr[i].reshape(-1))
                            print(f"[DBG]  + fallback lead[{i}] len={arr[i].reshape(-1).size}")
                    elif arr.ndim == 1:
                        series_list.append(arr.reshape(-1))
                        print(f"[DBG]  + fallback single lead len={arr.reshape(-1).size}")
                break

    print(f"[DBG] Extracted {len(series_list)} lead(s)")
    return series_list[:max_leads]


def _build_messages_from_sample(sample: dict, eos_token: str = "") -> dict:
    """Build chat-style messages using pre/post prompts and a composite ECG plot image.

    Assistant content is the provided CoT `answer`.
    """
    pre = (sample.get("pre_prompt") or "").strip()
    post = (sample.get("post_prompt") or "").strip()
    ans = (sample.get("answer") or "").strip()
    print("[DBG] Building messages ...")
    print(f"[DBG]  pre_prompt len={len(pre)} post_prompt len={len(post)} answer len={len(ans)}")

    # Append EOS token to answer if provided
    if eos_token:
        ans = ans + eos_token
        print("[DBG]  appended EOS token to answer")

    # Extract ECG series and make plot image
    leads = _extract_ecg_series_from_sample(sample, max_leads=12)
    try:
        img = _ecg_leads_to_pil(leads)
        try:
            w, h = img.size
            print(f"[DBG]  image size: {w}x{h}")
        except Exception:
            print("[DBG]  image created (size unknown)")
    except Exception as e:
        print(f"[DBG][ERR] Failed to render ECG image: {e}")
        raise

    # Include question details if present (ECG-QA specifics)
    question = sample.get("question")
    if question:
        pre_text = f"{pre}\n\nQuestion: {question}" if pre else f"Question: {question}"
    else:
        pre_text = pre

    user_text = "\n\n".join([p for p in [pre_text, post] if p])
    print(f"[DBG]  user_text len={len(user_text)}")
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
    print("[DBG] Messages built successfully")
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

    # Build training chat examples with images from ECG-QA train split
    print(f"[DBG] Args: max_samples={args.max_samples}, exclude_comparison={args.exclude_comparison}, preload_processed_data={args.preload_processed_data}")
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
