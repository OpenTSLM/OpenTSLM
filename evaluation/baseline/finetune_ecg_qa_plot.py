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
import wfdb

# Ensure project src/ is on sys.path so we can import time_series_datasets
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# Also add project root so we can import sibling modules when running script directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from time_series_datasets.ecg_qa.plot_example import get_ptbxl_ecg_path

# Prefer local import when running from evaluation/baseline, fall back to package path
try:
    from common_finetune_sft import run_sft  # when cwd is this folder or script path is used
except ModuleNotFoundError:
    from evaluation.baseline.common_finetune_sft import run_sft


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _downsample_to_100hz(ecg_data: np.ndarray, original_freq: int) -> np.ndarray:
    """Downsample ECG data to 100Hz"""
    if original_freq == 100:
        return ecg_data

    # Calculate downsampling factor
    downsample_factor = original_freq // 100

    # Downsample by taking every nth sample
    downsampled_data = ecg_data[::downsample_factor]

    return downsampled_data


def _load_ecg_data(ecg_id: int) -> np.ndarray:
    """Load ECG data for a given ECG ID using wfdb (same as groundtruth)."""
    ecg_path = get_ptbxl_ecg_path(ecg_id)

    if not os.path.exists(ecg_path + ".dat"):
        raise FileNotFoundError(f"ECG file not found: {ecg_path}.dat")

    # Read ECG data using wfdb - returns (samples, leads) shape
    ecg_data, meta = wfdb.rdsamp(ecg_path)

    return ecg_data


def _ecg_to_pil(ecg_data: np.ndarray) -> Image.Image:
    """Render ECG data to a PIL RGB image (identical to groundtruth create_ecg_plot)."""

    # Downsample to 100Hz if needed
    if ecg_data.shape[0] > 1000:  # Likely 500Hz data
        ecg_data = _downsample_to_100hz(ecg_data, 500)

    n = min(ecg_data.shape[1], 12)  # Up to 12 leads
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.0 * n), dpi=100)
    if n == 1:
        axes = [axes]

    # Create time array for 100Hz sampling (10 seconds)
    time_points = np.arange(0, 10, 0.01)  # 100Hz for 10 seconds

    for i in range(n):
        ax = axes[i]
        lead_name = LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"Lead {i+1}"

        # Plot the ECG signal for this lead - ecg_data is (samples, leads)
        ax.plot(time_points, ecg_data[:, i], linewidth=2, color="k", alpha=1.0)

        # Add grid lines (millimeter paper style)
        # Major grid lines (every 0.2s and 0.5mV)
        ax.vlines(
            np.arange(0, 10, 0.2), -2.5, 2.5, colors="r", alpha=0.3, linewidth=0.5
        )
        ax.hlines(
            np.arange(-2.5, 2.5, 0.5), 0, 10, colors="r", alpha=0.3, linewidth=0.5
        )

        # Minor grid lines (every 0.04s and 0.1mV)
        ax.vlines(
            np.arange(0, 10, 0.04), -2.5, 2.5, colors="r", alpha=0.1, linewidth=0.3
        )
        ax.hlines(
            np.arange(-2.5, 2.5, 0.1), 0, 10, colors="r", alpha=0.1, linewidth=0.3
        )

        ax.set_xticks(np.arange(0, 11, 1.0))
        ax.set_ylabel(f"Lead {lead_name} (mV)", fontweight="bold")
        ax.margins(0.0)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(f"Lead {lead_name}", fontweight="bold", pad=10)

    plt.tight_layout()

    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


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
    """Build chat-style messages with ECG plot image for training."""
    pre = (sample.get("pre_prompt") or "").strip()
    post = (sample.get("post_prompt") or "").strip()
    ans = (sample.get("answer") or "").strip()

    if eos_token:
        ans = ans + eos_token

    # Load raw ECG data from wfdb (same as groundtruth)
    ecg_id = _get_ecg_id_from_sample(sample)
    ecg_data = _load_ecg_data(ecg_id)
    img = _ecg_to_pil(ecg_data)

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
