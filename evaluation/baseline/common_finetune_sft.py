#!/usr/bin/env python3
"""
Common SFT (LoRA) fine-tuning helper for HF Image-Text-to-Text models.
Exposes run_sft(train_examples, **kwargs) where each example is a dict with a
"messages" field (chat template) and optional images embedded in the messages.

Dependencies:
  transformers, datasets, peft, trl, accelerate (and bitsandbytes if you want QLoRA)
"""
from __future__ import annotations
import io
import os
import sys
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import wfdb
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from peft import LoraConfig
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from PIL import Image

# Ensure project src/ is on sys.path so we can import time_series_datasets
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from time_series_datasets.ecg_qa.plot_example import get_ptbxl_ecg_path

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
    """Load ECG data for a given ECG ID using wfdb."""
    ecg_path = get_ptbxl_ecg_path(ecg_id)

    if not os.path.exists(ecg_path + ".dat"):
        raise FileNotFoundError(f"ECG file not found: {ecg_path}.dat")

    # Read ECG data using wfdb - returns (samples, leads) shape
    ecg_data, meta = wfdb.rdsamp(ecg_path)

    return ecg_data


def _ecg_to_image(ecg_data: np.ndarray) -> Image.Image:
    """Render ECG data to PIL Image (for lazy loading in collate_fn)."""

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

    # Return PIL Image directly
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    """Extract PIL images from chat messages. Handles lazy ecg_id, bytes, and PIL Images."""
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if not isinstance(element, dict):
                continue

            # Handle lazy ecg_id reference (render on-demand)
            ecg_id = element.get("ecg_id")
            if ecg_id is not None:
                ecg_data = _load_ecg_data(ecg_id)
                image = _ecg_to_image(ecg_data)
                image_inputs.append(image)
                continue

            # Handle pre-rendered bytes (backwards compatibility)
            image = element.get("image")
            if image is None:
                continue  # Text elements get "image": None from Dataset serialization

            # Handle bytes (PNG data) - convert to PIL Image
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))

            image_inputs.append(image.convert("RGB"))
    return image_inputs


def run_sft(
    train_examples: List[dict],
    *,
    output_dir: str,
    llm_id: str = "google/gemma-3-4b-pt",
    epochs: int = 1,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_seq_len: int = 4096,
    logging_steps: int = 10,
    save_steps: int = 100,  # Save less frequently to save disk space
    bf16: bool = True,
) -> None:
    """Run LoRA SFT on chat-style examples (with images) and save adapters.

    Args:
        train_examples: List of dicts, each containing a "messages" list compatible
            with the processor's chat template. Image elements should be PIL Images
            placed as dicts with {"type": "image", "image": PIL.Image}.
        output_dir: Where to save adapters and processor
        llm_id: HF model id (e.g., google/gemma-3-4b-pt)
        epochs, learning_rate, per_device_train_batch_size, gradient_accumulation_steps,
        max_seq_len: Usual training hyperparameters
        logging_steps, save_steps, bf16: Trainer settings
    """
    if not train_examples:
        raise ValueError("train_examples is empty; provide at least one training example")

    os.makedirs(output_dir, exist_ok=True)

    ds = Dataset.from_list(train_examples)

    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    model = AutoModelForImageTextToText.from_pretrained(
        llm_id,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    lora_cfg = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        bf16=bf16,
        report_to=[],
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=max_seq_len,
        packing=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        max_grad_norm=0.3,
    )

    def collate_fn(examples: List[dict]):
        """Collate chat examples into a batch with masked labels."""
        texts = []
        images = []
        for ex in examples:
            msgs = ex["messages"]
            text = processor.apply_chat_template(
                msgs, add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(process_vision_info(msgs))

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()

        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        special_map = processor.tokenizer.special_tokens_map
        boi_id = None
        if isinstance(special_map, dict) and "boi_token" in special_map:
            boi_id = processor.tokenizer.convert_tokens_to_ids(special_map["boi_token"])
        if boi_id is not None:
            labels[labels == boi_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    trainer = SFTTrainer(
        model=model,
        peft_config=lora_cfg,
        processing_class=processor,
        train_dataset=ds,
        args=training_args,
        data_collator=collate_fn,
    )

    resume_from_checkpoint = None
    import glob
    import os as os_module
    checkpoints = glob.glob(f"{output_dir}/checkpoint-*")
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        if os_module.path.exists(os_module.path.join(latest_checkpoint, "trainer_state.json")):
            resume_from_checkpoint = latest_checkpoint
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            print(f"Checkpoint {latest_checkpoint} is incomplete, starting from scratch")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved LoRA adapters and processor to: {output_dir}")

    del model
    del trainer
    torch.cuda.empty_cache()

