# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import json
import os
from typing import List
from opentslm.time_series_datasets.TSQADataset import TSQADataset
from opentslm.time_series_datasets.monash.MonashSPO2QADataset import MonashSPO2QADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model.projector.MLPProjector import MLPProjector
from opentslm.model_config import (
    BATCH_SIZE,
    EARLY_STOP_PAT,
    GRAD_CLIP_NORM,
    LR_ENCODER,
    LR_PROJECTOR,
    NUM_EPOCHS,
    PATCH_SIZE,
    RESULTS_FILE,
    WARMUP_FRAC,
    WEIGHT_DECAY,
)


# ---------------------------
# Device setup
# ---------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ---------------------------
# Model
# ---------------------------
encoder = TransformerCNNEncoder().to(device)
model = OpenTSLMSP(encoder=encoder, projector_class=MLPProjector, device=device).to(
    device
)


# — Freeze the LLM backbone so we only update encoder + projector
for p in model.llm.parameters():
    p.requires_grad = False

# Parameter groups with different learning rates
enc_params = list(model.encoder.parameters())
proj_params = list(model.projector.projector.parameters())
optimizer = AdamW(
    [
        {"params": enc_params, "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY},
        {"params": proj_params, "lr": LR_PROJECTOR, "weight_decay": WEIGHT_DECAY},
    ]
)


def merge_data_loaders(
    datasets: List[Dataset], shuffle: bool, batch_size: int, patch_size: int
) -> DataLoader:
    merged_ds = ConcatDataset(datasets)
    return DataLoader(
        merged_ds,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=patch_size
        ),
    )


QA_DATASET_CLASSES = [TSQADataset]

# ---------------------------
# Data loaders
# ---------------------------
train_loader = merge_data_loaders(
    [
        dataset_class(
            "train",
            EOS_TOKEN=model.get_eos_token(),
        )
        for dataset_class in QA_DATASET_CLASSES
    ],
    shuffle=True,
    batch_size=BATCH_SIZE,
    patch_size=PATCH_SIZE,
)

val_loader = merge_data_loaders(
    [
        dataset_class(
            "validation",
            EOS_TOKEN=model.get_eos_token(),
        )
        for dataset_class in QA_DATASET_CLASSES
    ],
    shuffle=False,
    batch_size=1,
    patch_size=PATCH_SIZE,
)
test_loader = merge_data_loaders(
    [
        dataset_class(
            "test",
            EOS_TOKEN=model.get_eos_token(),
        )
        for dataset_class in QA_DATASET_CLASSES
    ],
    shuffle=False,
    batch_size=1,
    patch_size=PATCH_SIZE,
)


# Scheduler (linear warmup + decay)
TOTAL_STEPS = NUM_EPOCHS * len(train_loader)
WARMUP_STEPS = int(WARMUP_FRAC * TOTAL_STEPS)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=TOTAL_STEPS,
)

# ---------------------------
# Helpers
# ---------------------------


def _save_best(epoch: int, val_loss: float):
    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "projector_state": model.projector.state_dict(),
            "val_loss": val_loss,
            "epoch": epoch,
        },
        "best_encoder.pt",
    )


def _load_best():
    if os.path.exists("best_encoder.pt"):
        ckpt = torch.load("best_encoder.pt", map_location=device)
        model.encoder.load_state_dict(ckpt["encoder_state"])
        model.projector.load_state_dict(ckpt["projector_state"])
        return ckpt.get("epoch", "?")
    return None


def _evaluate_test():
    """Run best model on test set and write prompt+generation+gold to JSONL."""
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test inference"):
            # batch is a List[Dict], same as in compute_loss/generate
            gens = model.generate(batch)  # returns List[str] of length len(batch)

            # collect each sample’s I/O
            for sample, gen in zip(batch, gens):
                results.append(
                    {
                        "pre_prompt": sample["pre_prompt"],
                        "time_series_text": sample["time_series_text"],
                        "post_prompt": sample["post_prompt"],
                        "generated": gen,
                        "gold": sample["answer"],
                    }
                )

    # write JSONL
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅  Test predictions saved to {RESULTS_FILE} (n={len(results)})")


# ---------------------------
# Training loop with early stopping
# ---------------------------


def train():
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        running_loss = 0.0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for batch in prog:
            optimizer.zero_grad()
            # batch is List[PromptWithAnswer]
            loss = model.compute_loss(batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            prog.set_postfix(
                loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

        avg_train_loss = running_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch} — train loss: {avg_train_loss:.4f}")

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                val_loss += model.compute_loss(batch).item()
        avg_val_loss = val_loss / len(val_loader)
        tqdm.write(f"Epoch {epoch} — val   loss: {avg_val_loss:.4f}\n")

        # Early stopping
        if avg_val_loss + 1e-4 < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            _save_best(epoch, avg_val_loss)
            tqdm.write("✔️  New best model saved.\n")

        else:
            epochs_no_improve += 1
            tqdm.write(
                f"No improvement for {epochs_no_improve}/{EARLY_STOP_PAT} epochs."
            )
            if epochs_no_improve >= EARLY_STOP_PAT:
                tqdm.write("\nEarly stopping triggered.")
                break

    tqdm.write("Training finished.\n")

    # Test evaluation
    best_epoch = _load_best()
    if best_epoch is not None:
        print(f"Loaded best checkpoint from epoch {best_epoch} for test evaluation.")
    _evaluate_test()


if __name__ == "__main__":
    train()
