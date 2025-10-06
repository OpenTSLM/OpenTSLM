#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import json
import os as _os
import argparse
from typing import List, Optional, Dict, Any, Callable
from time_series_datasets.TSQADataset import TSQADataset
from time_series_datasets.m4.M4QADataset import M4QADataset
from time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from time_series_datasets.ngafid_cot.NGAFIDCoTQADataset import NGAFIDCoTQADataset
from time_series_datasets.fault_detection_cot.FaultDetectionCoTQADataset import FaultDetectionCoTQADataset
from time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    FullStateDictConfig,
    StateDictType,
)
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from model.llm.OpenTSLMSP import OpenTSLMSP
from model.projector.MLPProjector import MLPProjector
import datetime
from logger import get_logger, set_global_verbose

from model_config import (
    BATCH_SIZE,
    EARLY_STOP_PAT,
    GRAD_CLIP_NORM,
    LR_ENCODER,
    LR_PROJECTOR,
    NUM_EPOCHS,
    PATCH_SIZE,
    WARMUP_FRAC,
    WEIGHT_DECAY,
)


# Global stage configuration - users can modify this to mix and match stages
CURRICULUM_STAGES = [
    "stage1_mcq",
    "stage2_captioning",
    "stage3_cot",
    "stage4_sleep_cot",
    "stage5_ecg_cot",
    "stage6_ngafid_cot",
    "stage7_fault_detection_cot",
]


class CurriculumTrainer:
    """
    Curriculum learning trainer for OpenTSLM models.
    Trains models stage by stage with shared training logic.
    While this may look like a lot of code, it's actually quite modular.
    We simply train either OpenTSLMSP or OpenTSLMFlamingo, both using the same training loop.
    We train across different stages:
    - stage1_mcq: Trains the model on a time-series MCQ dataset (TSQA)
    - stage2_captioning: Trains the model on a time-series captioning dataset (M4 time series captioning)
    - stage3_cot: Trains the model on a chain-of-thought reasoning dataset (HAR CoT)
    - stage4_sleep_cot: Trains the model on sleep stage classification with chain-of-thought reasoning
    - stage5_ecg_cot: Trains the model on ECG QA with chain-of-thought reasoning
    - stage6_ngafid_cot: Trains the model on predictive maintenance reasoning (NGAFID CoT)

    Features:
    - Automatic loss history tracking saved to loss_history.txt in each stage's checkpoints directory
    - Loss history is appended to when resuming training, preserving all previous epochs
    - Displays previous loss history when resuming training

    If you run this script, you should be able to reproduce our results from the paper.
    All datasets are automatically downloaded and processed.
    """

    def _sanitize_llm_id(self, llm_id: str) -> str:
        """Sanitize llm_id for use in directory names (e.g., meta-llama/Llama-3.2-1B -> Llama3_2_1B)"""
        if not llm_id:
            return "unknown_llm"
        # Take last part after /, replace . and - with _
        name = llm_id.split("/")[-1]
        name = name.replace(".", "_").replace("-", "_")
        # Optionally, remove duplicate underscores
        while "__" in name:
            name = name.replace("__", "_")
        return name

    def __init__(
        self,
        model_type: str,
        device: str = None,
        gradient_checkpointing: bool = False,
        dist_url: str = "env://",
        dist_backend: str = "nccl",
        local_rank: int = int(os.environ.get("LOCAL_RANK", 0)),
        llm_id: str = None,
    ):
        """
        Initialize the curriculum trainer.

        Args:
            model_type: Either 'OpenTSLMSP' or 'OpenTSLMFlamingo'
            device: Device to use for training ('cuda', 'mps', or 'cpu')
            gradient_checkpointing: Enable gradient checkpointing
            dist_url: URL used to set up distributed training
            dist_backend: Distributed backend
            local_rank: Local GPU rank
            llm_id: LLM model ID (e.g., 'google/medgemma-2b', 'meta-llama/Llama-3.2-1B')
        """
        self.model_type = model_type
        self.device = device or self._get_device()
        if self.device == "mps":
            print(
                "🚨 Warning: Using MPS, might not be fully compatible with the model. Use CUDA for best results."
            )
        self.llm_id = llm_id
        self.llm_id_safe = self._sanitize_llm_id(llm_id)

        # Distributed training parameters
        self.gradient_checkpointing = gradient_checkpointing
        self.dist_url = dist_url
        self.dist_backend = dist_backend
        self.local_rank = local_rank

        # Initialize distributed training if needed
        self.rank = 0
        self.world_size = 1
        if self._should_use_distributed():
            self._init_distributed()

        self.model = self._initialize_model()
        self.results_dir = os.path.join("results", self.llm_id_safe, self.model_type)
        self._create_results_dir()

    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _initialize_model(self):
        """Initialize the specified model type."""
        if self.model_type == "OpenTSLMSP":
            model = OpenTSLMSP(llm_id=self.llm_id, device=self.device).to(self.device)

        elif self.model_type == "OpenTSLMFlamingo":
            model = OpenTSLMFlamingo(
                cross_attn_every_n_layers=1,
                gradient_checkpointing=self.gradient_checkpointing,
                llm_id=self.llm_id,
                device=self.device,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Use DDP for multi-GPU training (simpler and than FSDP)
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            )
            if self.rank == 0:
                print(f"Wrapped {self.model_type} with DDP for distributed training")

        return model

    def _get_cast_dtype(self, precision: str):
        """Get cast dtype for mixed precision."""
        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        else:
            return None

    def _create_results_dir(self):
        """Create the results directory structure."""
        os.makedirs(self.results_dir, exist_ok=True)
        # model_dir now includes llm_id_safe
        model_dir = self.results_dir
        os.makedirs(model_dir, exist_ok=True)

        # Create stage directories based on global configuration
        for stage in CURRICULUM_STAGES:
            stage_dir = os.path.join(model_dir, stage)
            os.makedirs(stage_dir, exist_ok=True)
            os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(stage_dir, "results"), exist_ok=True)

    def _get_optimizer(
        self,
        batch_size: int = None,
        lr_encoder: float = None,
        lr_projector: float = None,
        lr_base: float = None,
    ):
        """Get optimizer for the model with configurable learning rates."""
        # Get the underlying model (handles DDP wrapping)
        model = self._get_model()

        if self.model_type == "OpenTSLMSP":
            # Parameter groups with different learning rates for SP
            enc_params = list(model.encoder.parameters())
            proj_params = list(model.projector.projector.parameters())

            # Use provided learning rates or defaults
            encoder_lr = lr_encoder if lr_encoder is not None else LR_ENCODER
            projector_lr = lr_projector if lr_projector is not None else LR_PROJECTOR

            param_groups = [
                {"params": enc_params, "lr": encoder_lr, "weight_decay": WEIGHT_DECAY},
                {
                    "params": proj_params,
                    "lr": projector_lr,
                    "weight_decay": WEIGHT_DECAY,
                },
            ]

            # Add LoRA parameters if enabled
            if hasattr(model, "lora_enabled") and model.lora_enabled:
                lora_params = model.get_lora_parameters()
                if lora_params:
                    # Use projector LR for LoRA parameters (similar fine-tuning nature)
                    param_groups.append(
                        {
                            "params": lora_params,
                            "lr": projector_lr,
                            "weight_decay": WEIGHT_DECAY,
                        }
                    )
                    if self.rank == 0:
                        print(f"📊 Learning rates for {self.model_type} (with LoRA):")
                        print(f"   Encoder LR: {encoder_lr:.2e}")
                        print(f"   Projector LR: {projector_lr:.2e}")
                        print(
                            f"   LoRA LR: {projector_lr:.2e} ({len(lora_params)} parameters)"
                        )
                else:
                    raise RuntimeError(
                        "LoRA is enabled but no trainable LoRA parameters found. This indicates a LoRA configuration issue."
                    )
            else:
                if self.rank == 0:
                    print(f"📊 Learning rates for {self.model_type}:")
                    print(f"   Encoder LR: {encoder_lr:.2e}")
                    print(f"   Projector LR: {projector_lr:.2e}")

            return AdamW(param_groups)
        else:
            # For Flamingo, use grouped parameters
            params_to_optimize = model.named_parameters()
            params_to_optimize = list(
                filter(
                    lambda x: x[1].requires_grad
                    and not getattr(x[1], "exclude_from_optimizer", False),
                    params_to_optimize,
                )
            )

            # Group parameters for weight decay
            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                if "gated_cross_attn" in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)

            # Use provided base learning rate or default
            base_lr = lr_base if lr_base is not None else 2e-4

            if self.rank == 0:
                print(f"📊 Learning rate for {self.model_type}:")
                print(f"   Base LR: {base_lr:.2e}")

            return torch.optim.AdamW(
                [
                    {"params": params_with_wd, "weight_decay": 0.1},
                    {"params": params_without_wd, "weight_decay": 0.0},
                ],
                lr=base_lr,
            )

    def _merge_data_loaders(
        self,
        datasets: List[Dataset],
        shuffle: bool,
        batch_size: int,
        patch_size: int,
        distribute_data: bool = False,
    ) -> DataLoader:
        """Create a merged data loader from multiple datasets."""
        merged_ds = ConcatDataset(datasets)

        # Use distributed sampler if distributed training is enabled
        if distribute_data and dist.is_initialized():
            sampler = DistributedSampler(
                merged_ds, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle
            )
            return DataLoader(
                merged_ds,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=patch_size
                ),
            )
        else:
            return DataLoader(
                merged_ds,
                shuffle=shuffle,
                batch_size=batch_size,
                collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=patch_size
                ),
            )

    def _save_checkpoint(
        self, stage: str, epoch: int, val_loss: float, optimizer, scheduler
    ):
        """Save model checkpoint for a specific stage."""
        checkpoint_dir = os.path.join(self.results_dir, stage, "checkpoints")

        # Only save on rank 0 for distributed training
        if dist.is_initialized() and self.rank != 0:
            return

        # Get the underlying model (handles DDP wrapping)
        model = self._get_model()

        if self.model_type == "OpenTSLMSP":
            checkpoint = {
                "encoder_state": model.encoder.state_dict(),
                "projector_state": model.projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }

            # Add LoRA state to checkpoint
            model.save_lora_state_to_checkpoint(checkpoint)
        else:
            # Handle DDP or single GPU case for OpenTSLMFlamingo
            model_state = model.state_dict()
            if hasattr(self.model, "module"):
                # Remove 'module.' prefix for DDP
                model_state = {
                    k.replace("module.", ""): v for k, v in model_state.items()
                }
            checkpoint = {
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }

        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

        # Check disk space before saving
        if self.rank == 0:
            import shutil

            total, used, free = shutil.disk_usage(checkpoint_dir)
            free_gb = free / (1024**3)
            print(f"💾 Disk space: {free_gb:.2f} GB free in {checkpoint_dir}")

            # Estimate checkpoint size (rough estimate)
            estimated_size_gb = sum(
                p.numel() * p.element_size() for p in self._get_model().parameters()
            ) / (1024**3)
            if (
                free_gb < estimated_size_gb * 2
            ):  # Need at least 2x the size for safe writing
                print(
                    f"⚠️  Warning: Low disk space. Need ~{estimated_size_gb:.2f} GB, have {free_gb:.2f} GB free"
                )

        # Try to save with error handling
        try:
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            if self.rank == 0:
                print(f"❌ Failed to save checkpoint: {e}")
                print(f"   Checkpoint path: {checkpoint_path}")
                print(
                    f"   Checkpoint size: {sum(p.numel() * p.element_size() for p in self._get_model().parameters()) / 1024**3:.2f} GB"
                )

                raise RuntimeError(f"Failed to save checkpoint: {e}")

    def _save_loss_history(
        self, stage: str, epoch: int, train_loss: float, val_loss: float
    ):
        """Save loss history to a file for tracking training progress."""
        if dist.is_initialized() and self.rank != 0:
            return  # Only save on rank 0 for distributed training

        checkpoint_dir = os.path.join(self.results_dir, stage, "checkpoints")
        loss_history_file = os.path.join(checkpoint_dir, "loss_history.txt")

        # Ensure the directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create the file with header if it doesn't exist
        if not os.path.exists(loss_history_file):
            with open(loss_history_file, "w") as f:
                f.write("Epoch\tTrain_Loss\tVal_Loss\n")
                f.write("-" * 30 + "\n")

        # Append the current epoch's losses
        with open(loss_history_file, "a") as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")

    def _display_loss_history(self, stage: str):
        """Display the loss history for a stage if available."""
        if dist.is_initialized() and self.rank != 0:
            return  # Only display on rank 0 for distributed training

        checkpoint_dir = os.path.join(self.results_dir, stage, "checkpoints")
        loss_history_file = os.path.join(checkpoint_dir, "loss_history.txt")

        if os.path.exists(loss_history_file):
            try:
                with open(loss_history_file, "r") as f:
                    lines = f.readlines()

                if len(lines) > 2:  # More than just header
                    print(f"📊 Previous loss history for {stage}:")
                    print("   Epoch\tTrain_Loss\tVal_Loss")
                    print("   " + "-" * 30)

                    # Show last 5 epochs (or all if less than 5)
                    start_idx = max(2, len(lines) - 5)  # Skip header lines
                    for line in lines[start_idx:]:
                        if line.strip() and not line.startswith("-"):
                            parts = line.strip().split("\t")
                            if len(parts) == 3:
                                epoch, train_loss, val_loss = parts
                                print(f"   {epoch}\t{train_loss}\t{val_loss}")

                    if len(lines) > 7:  # More than 5 epochs
                        print(f"   ... and {len(lines) - 7} more epochs")
                    print()
            except Exception as e:
                print(f"⚠️  Could not read loss history: {e}")

    def _load_checkpoint(
        self, stage: str, optimizer, scheduler, eval_only: bool = False
    ):
        """Load model checkpoint for a specific stage."""
        checkpoint_path = os.path.join(
            self.results_dir, stage, "checkpoints", "best_model.pt"
        )

        if os.path.exists(checkpoint_path):
            # Always load checkpoint to CPU first to avoid GPU OOM spikes
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

            # Get the underlying model (handles DDP wrapping)
            model = self._get_model()

            if self.model_type == "OpenTSLMSP":
                model.encoder.load_state_dict(checkpoint["encoder_state"])
                model.projector.load_state_dict(checkpoint["projector_state"])

                # Load LoRA state using the OpenTSLMSP method (allow missing for backward compatibility)
                try:
                    model.load_lora_state_from_checkpoint(
                        checkpoint, allow_missing=True
                    )
                except RuntimeError as e:
                    if self.rank == 0:
                        print(f"❌ Failed to load LoRA state from checkpoint: {e}")
                    raise

                # Only load optimizer state when training
                if (
                    not eval_only
                    and optimizer is not None
                    and "optimizer_state" in checkpoint
                ):
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
            else:
                # Handle DDP or single GPU case for OpenTSLMFlamingo
                model_state = checkpoint["model_state"]
                if hasattr(self.model, "module"):
                    # Add 'module.' prefix for DDP
                    model_state = {f"module.{k}": v for k, v in model_state.items()}

                # Load state dict with strict=False to handle missing keys
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        model_state, strict=False
                    )
                    if missing_keys and self.rank == 0:
                        print(
                            f"⚠️  Warning: Missing keys when loading checkpoint for {stage}:"
                        )
                        for key in missing_keys[:10]:  # Show first 10 missing keys
                            print(f"   - {key}")
                        if len(missing_keys) > 10:
                            print(f"   ... and {len(missing_keys) - 10} more keys")
                    if unexpected_keys and self.rank == 0:
                        print(
                            f"⚠️  Warning: Unexpected keys when loading checkpoint for {stage}:"
                        )
                        for key in unexpected_keys[
                            :10
                        ]:  # Show first 10 unexpected keys
                            print(f"   - {key}")
                        if len(unexpected_keys) > 10:
                            print(f"   ... and {len(unexpected_keys) - 10} more keys")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load model state from checkpoint for {stage}: {e}"
                    )

                # Only load optimizer state when training
                if (
                    not eval_only
                    and optimizer is not None
                    and "optimizer_state" in checkpoint
                ):
                    optimizer.load_state_dict(checkpoint["optimizer_state"])

            # Only load scheduler state when training
            if (
                not eval_only
                and scheduler is not None
                and "scheduler_state" in checkpoint
            ):
                scheduler.load_state_dict(checkpoint["scheduler_state"])

            return checkpoint.get("epoch", "?"), checkpoint.get(
                "val_loss", float("inf")
            )
        return None, float("inf")

    def _load_previous_stage_model(
        self, current_stage: str
    ) -> Optional[Dict[str, Any]]:
        """Load the best model from the previous stage and return its metrics."""
        try:
            current_idx = CURRICULUM_STAGES.index(current_stage)
            if current_idx == 0:
                # First stage, no previous model to load
                return None
            previous_stage = CURRICULUM_STAGES[current_idx - 1]
            metrics_file = os.path.join(
                self.results_dir, previous_stage, "results", "metrics.json"
            )
            if not os.path.exists(metrics_file):
                # PATCH: If running stage2_captioning and previous stage metrics are missing, skip loading
                if current_stage == "stage2_captioning":
                    if self.rank == 0:
                        print(
                            f"⚠️  Skipping previous stage {previous_stage} because metrics file not found: {metrics_file}"
                        )
                    return None
                raise RuntimeError(
                    f"Previous stage {previous_stage} metrics file not found: {metrics_file}"
                )
            # Be robust to malformed JSON (e.g., concurrent writes or concatenated JSON)
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
            except Exception as e:
                if self.rank == 0:
                    print(
                        f"⚠️  Warning: Could not parse metrics file for {previous_stage} ({metrics_file}): {e}"
                    )
                    print("   Proceeding without previous metrics.")
                metrics = {}
            # Load the model weights from previous stage
            checkpoint_path = os.path.join(
                self.results_dir, previous_stage, "checkpoints", "best_model.pt"
            )
            if not os.path.exists(checkpoint_path):
                # PATCH: If running stage2_captioning and previous stage checkpoint is missing, skip loading
                if current_stage == "stage2_captioning":
                    if self.rank == 0:
                        print(
                            f"⚠️  Skipping previous stage {previous_stage} because checkpoint not found: {checkpoint_path}"
                        )
                    return None
                raise RuntimeError(
                    f"Previous stage {previous_stage} checkpoint not found: {checkpoint_path}"
                )
            print(
                "Loading checkpoint from previous stage: ",
                checkpoint_path,
                "and model type: ",
                self.model_type,
                "and llm_id: ",
                self.llm_id,
            )
            print("This might take a while")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            # Get the underlying model (handles DDP wrapping)
            model = self._get_model()
            if self.model_type == "OpenTSLMSP":
                model.encoder.load_state_dict(checkpoint["encoder_state"])
                model.projector.load_state_dict(checkpoint["projector_state"])

                # Load LoRA state from previous stage (allow missing for stage transitions)
                try:
                    loaded_count = model.load_lora_state_from_checkpoint(
                        checkpoint, allow_missing=True
                    )
                    if loaded_count > 0 and self.rank == 0:
                        print(
                            f"📥 Loaded LoRA adapters from previous stage: {loaded_count} parameters"
                        )
                except RuntimeError as e:
                    if self.rank == 0:
                        print(f"❌ Failed to load LoRA state from previous stage: {e}")
                    # For previous stage loading, we can be more tolerant of LoRA mismatches
                    # as stages might have different LoRA configurations
            else:
                # Handle OpenTSLMFlamingo with graceful loading
                model_state = checkpoint["model_state"]
                if hasattr(self.model, "module"):
                    # Add 'module.' prefix for DDP
                    model_state = {f"module.{k}": v for k, v in model_state.items()}
                # Load state dict with strict=False to handle missing keys
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        model_state, strict=False
                    )
                    if missing_keys and self.rank == 0:
                        print(
                            f"⚠️  Warning: Missing keys when loading previous stage {previous_stage}:"
                        )
                        for key in missing_keys[:5]:  # Show first 5 missing keys
                            print(f"   - {key}")
                        if len(missing_keys) > 5:
                            print(f"   ... and {len(missing_keys) - 5} more keys")
                        print(
                            f"   This is normal when transitioning between stages with different model configurations."
                        )
                    if unexpected_keys and self.rank == 0:
                        print(
                            f"⚠️  Warning: Unexpected keys when loading previous stage {previous_stage}:"
                        )
                        for key in unexpected_keys[:5]:  # Show first 5 unexpected keys
                            print(f"   - {key}")
                        if len(unexpected_keys) > 5:
                            print(f"   ... and {len(unexpected_keys) - 5} more keys")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load model state from previous stage {previous_stage}: {e}"
                    )
            return {
                "stage": previous_stage,
                "metrics": metrics,
                "epoch": checkpoint.get("epoch", "?"),
                "val_loss": checkpoint.get("val_loss", "?"),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load previous stage model: {e}")

    def _calculate_accuracy(
        self, predictions: List[str], gold_answers: List[str]
    ) -> float:
        """Calculate accuracy for MCQ tasks."""
        correct = 0
        total = len(predictions)

        for pred, gold in zip(predictions, gold_answers):
            # Clean up predictions and gold answers
            pred_clean = pred.strip()
            gold_clean = gold.strip()

            # Check if gold starts with the cleaned prediction (more robust matching)
            if gold_clean.startswith(pred_clean) or pred_clean == gold_clean:
                correct += 1

        return correct / total if total > 0 else 0.0

    def _evaluate_stage(
        self,
        stage: str,
        test_loader: DataLoader,
        stage_name: str,
        metric_func: Callable = None,
        epoch: int = None,
    ) -> Dict[str, Any]:
        """Evaluate model on test set for a specific stage."""
        # Enable eval mode for all ranks
        self.model.eval()
        results = []
        test_loss = 0.0

        # Set higher max_tokens for generation during evaluation
        max_new_tokens = 2000

        # Prepare per-rank streaming writer for test predictions
        results_file_rank = os.path.join(
            self.results_dir,
            stage_name,
            "results",
            f"test_predictions_rank_{self.rank if dist.is_initialized() else 0}.jsonl",
        )
        final_results_file = os.path.join(
            self.results_dir, stage_name, "results", "test_predictions.jsonl"
        )
        results_fp = None
        # Ensure directory exists (defensive)
        os.makedirs(os.path.dirname(results_file_rank), exist_ok=True)
        if self.rank == 0:
            print(f"[Eval] rank={self.rank}, world_size={self.world_size}")
            print(f"Saving per-rank test predictions to: {results_file_rank}")
            if dist.is_initialized():
                print(
                    f"Final merged predictions will be saved to: {final_results_file}"
                )
        # Open per-rank file in write mode to start fresh, then append per-sample
        results_fp = open(results_file_rank, "w", encoding="utf-8")
        if not results_fp:
            raise RuntimeError(
                f"Failed to open per-rank results file: {results_file_rank}"
            )
        try:
            with torch.no_grad():
                for batch in tqdm(
                    test_loader, desc=f"Evaluating {stage_name}", disable=self.rank != 0
                ):
                    # Generate predictions with higher max_tokens (skip separate loss computation)
                    predictions = self._get_model().generate(
                        batch, max_new_tokens=max_new_tokens
                    )

                    # Collect results
                    for sample, pred in zip(batch, predictions):
                        result = {
                            "pre_prompt": sample["pre_prompt"],
                            "time_series_text": sample["time_series_text"],
                            "post_prompt": sample["post_prompt"],
                            "generated": pred,
                            "gold": sample["answer"],
                        }

                        # Attach stable identifiers when available so downstream tools can match
                        if "sample_id" in sample and sample["sample_id"] is not None:
                            result["sample_id"] = sample["sample_id"]
                        if "unique_sample_id" in sample and sample["unique_sample_id"] is not None:
                            result["unique_sample_id"] = sample["unique_sample_id"]
                        if "flight_id_before" in sample and sample["flight_id_before"] is not None:
                            result["flight_id_before"] = sample["flight_id_before"]

                        # Add time series ID for stage2 captioning
                        if stage == "stage2_captioning" and "id" in sample:
                            result["time_series_id"] = sample["id"]

                        # Add template_id and ecg_id for stage5_ecg_cot
                        if stage == "stage5_ecg_cot":
                            if "template_id" in sample:
                                result["template_id"] = sample["template_id"]
                            if "ecg_id" in sample:
                                result["ecg_id"] = sample["ecg_id"]
                            if "correct_answer" in sample:
                                result["correct_answer"] = sample["correct_answer"]

                        # Attach identifiers for fault detection mapping
                        if stage == "stage7_fault_detection_cot":
                            if "sample_id" in sample and sample["sample_id"] is not None:
                                result["sample_id"] = sample["sample_id"]
                            if "question" in sample:
                                result["question"] = sample["question"]
                            if "answer_label" in sample:
                                result["answer_label"] = sample["answer_label"]
                            if "numeric_label" in sample:
                                result["numeric_label"] = sample["numeric_label"]
                            if "template_id" in sample:
                                result["template_id"] = sample["template_id"]
                            if "fault_description" in sample:
                                result["fault_description"] = sample["fault_description"]

                        results.append(result)
                        # Stream write each result immediately to per-rank file
                        results_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                        results_fp.flush()
                        try:
                            os.fsync(results_fp.fileno())
                        except Exception:
                            pass
        finally:
            if results_fp is not None:
                results_fp.close()

        # Synchronize all ranks before merging
        if dist.is_initialized():
            dist.barrier()

        # Rank 0 merges per-rank files into final results file
        if (not dist.is_initialized()) or (self.rank == 0):
            try:
                # Overwrite final file each evaluation
                with open(final_results_file, "w", encoding="utf-8") as merged_fp:
                    if dist.is_initialized():
                        num_ranks = self.world_size
                    else:
                        num_ranks = 1
                    for r in range(num_ranks):
                        part_file = os.path.join(
                            self.results_dir,
                            stage_name,
                            "results",
                            f"test_predictions_rank_{r}.jsonl",
                        )
                        if os.path.exists(part_file):
                            with open(part_file, "r", encoding="utf-8") as pf:
                                for line in pf:
                                    merged_fp.write(line)
                if self.rank == 0:
                    print(f"Merged per-rank predictions into: {final_results_file}")
            finally:
                pass

        # Report test loss as NaN since we skip explicit loss computation during evaluation
        # Before, we were computing the loss explicitly, but this required to run the model twice, once for loss and once for predictions.
        avg_test_loss = float("nan")

        # Calculate stage-specific metrics
        metrics = {"test_loss": avg_test_loss}
        if epoch is not None:
            metrics["epoch"] = epoch
        if metric_func:
            # Compute metrics on rank 0 after merging, else minimal metrics
            if (not dist.is_initialized()) or (self.rank == 0):
                predictions = []
                gold_answers = []
                # Read from final merged file
                merged_path = final_results_file
                with open(merged_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            predictions.append(obj.get("generated", ""))
                            gold_answers.append(obj.get("gold", ""))
                        except Exception:
                            continue
                additional_metrics = metric_func(predictions, gold_answers)
                metrics.update(additional_metrics)

        # Save results only on rank 0 (or when not distributed)
        if (not dist.is_initialized()) or (self.rank == 0):
            # Save metrics
            metrics_file = os.path.join(
                self.results_dir, stage_name, "results", "metrics.json"
            )
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"✅ {stage_name} evaluation complete:")
            print(f"   Test predictions saved to: {final_results_file}")
            print(f"   Metrics saved to: {metrics_file}")
            print(f"   Max tokens used for generation: {max_new_tokens}")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")

        # Signal other ranks that evaluation is complete
        if dist.is_initialized():
            dist.barrier()

        return metrics

    def _is_evaluation_completed(self, stage: str) -> bool:
        """Check if evaluation was completed for a stage by looking for test predictions file."""
        test_predictions_file = os.path.join(
            self.results_dir, stage, "results", "test_predictions.jsonl"
        )
        metrics_file = os.path.join(self.results_dir, stage, "results", "metrics.json")

        # Check if both files exist
        if not os.path.exists(test_predictions_file) or not os.path.exists(
            metrics_file
        ):
            return False

        # Also check if metrics file has evaluation results
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            return "test_loss" in metrics
        except:
            return False

    def _train_stage(
        self,
        stage_name: str,
        dataset_class,
        num_epochs: int,
        lr_encoder: float,
        lr_projector: float,
        lr_base: float,
        metric_func: Callable = None,
        batch_size: int = None,
        eval_only: bool = False,
        sampler=None,
    ) -> Dict[str, Any]:
        """Generic training function for any stage."""
        epoch = None
        # Use provided batch_size or default to global BATCH_SIZE
        if batch_size is None:
            batch_size = BATCH_SIZE

        if self.rank == 0:
            print(f"\n🚀 Starting {stage_name} Training with {self.model_type}")
            if eval_only:
                print("🔍 EVAL-ONLY MODE: Skipping training, only running evaluation")
            print("=" * 60)
            print(f"📊 Stage Configuration:")
            print(f"   Epochs: {num_epochs}")
            if self.model_type == "OpenTSLMSP":
                print(f"   Encoder LR: {lr_encoder:.2e}")
                print(f"   Projector LR: {lr_projector:.2e}")
            else:
                print(f"   Base LR: {lr_base:.2e}")
            print(f"   Batch size per GPU: {batch_size}")
            if self.world_size > 1:
                print(f"   Effective batch size: {batch_size * self.world_size}")
            print()

        # Check if checkpoint exists when in eval_only mode
        if eval_only and not self._checkpoint_exists(stage_name):
            raise RuntimeError(
                f"Eval-only mode requires a checkpoint for {stage_name}, but none found at {os.path.join(self.results_dir, stage_name, 'checkpoints', 'best_model.pt')}"
            )

        # Load previous stage model and display metrics
        try:
            previous_stage_info = self._load_previous_stage_model(stage_name)
            if previous_stage_info:
                if self.rank == 0:
                    print(f"📂 Loading best model from {previous_stage_info['stage']}:")
                    print(f"   Achieved at epoch: {previous_stage_info['epoch']}")
                    val_loss = previous_stage_info["val_loss"]
                    if isinstance(val_loss, (int, float)):
                        print(f"   Validation loss: {val_loss:.4f}")
                    else:
                        print(f"   Validation loss: {val_loss}")
                    for metric, value in previous_stage_info["metrics"].items():
                        if isinstance(value, (int, float)):
                            print(f"   {metric}: {value:.4f}")
                        else:
                            print(f"   {metric}: {value}")
                    print()
            else:
                # Only allow fresh model for first stage
                if stage_name != CURRICULUM_STAGES[0]:
                    raise RuntimeError(
                        f"Cannot start {stage_name} with fresh model. Previous stage {CURRICULUM_STAGES[CURRICULUM_STAGES.index(stage_name) - 1]} must be completed first."
                    )
                if self.rank == 0:
                    print("🆕 Starting with fresh model (first stage)")
                    print()
        except Exception as e:
            if self.rank == 0:
                print(f"❌ Error loading previous stage: {e}")
            raise Exception(f"Error loading previous stage: {e}")

        # Check if evaluation was already completed
        evaluation_completed = self._is_evaluation_completed(stage_name)
        if evaluation_completed and self.rank == 0:
            print(
                f"✅ Evaluation already completed for {stage_name}, skipping training and evaluation"
            )
            print(f"📂 Loading existing metrics...")

            # Load and return existing metrics
            metrics_file = os.path.join(
                self.results_dir, stage_name, "results", "metrics.json"
            )
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            print(f"📊 Existing results for {stage_name}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")

            return metrics

        # Enable LoRA if needed for this stage
        self._enable_lora_if_needed(stage_name)

        # Initialize optimizer and scheduler
        optimizer = self._get_optimizer(batch_size, lr_encoder, lr_projector, lr_base)

        # Create data loaders
        if sampler is not None:
            if self.world_size > 1:
                get_logger().warning(
                    "BalancedBatchSampler was provided, but distributed training (DDP) is enabled. BalancedBatchSampler will NOT be used. Data will be sharded using DistributedSampler instead. Typically for stage3_cot it is better to use BalancedBatchSampler, if dataset is imbalanced."
                )
                train_loader = self._merge_data_loaders(
                    [
                        dataset_class(
                            "train", EOS_TOKEN=self._get_model().get_eos_token()
                        )
                    ],
                    shuffle=True,
                    batch_size=batch_size,
                    patch_size=PATCH_SIZE,
                    distribute_data=True,
                )
            else:
                train_dataset = dataset_class(
                    "train", EOS_TOKEN=self._get_model().get_eos_token()
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_sampler=sampler,
                    collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                        batch, patch_size=PATCH_SIZE
                    ),
                )
        else:
            train_loader = self._merge_data_loaders(
                [dataset_class("train", EOS_TOKEN=self._get_model().get_eos_token())],
                shuffle=True,
                batch_size=batch_size,
                patch_size=PATCH_SIZE,
                distribute_data=self.world_size > 1,
            )

        val_loader = self._merge_data_loaders(
            [dataset_class("validation", EOS_TOKEN=self._get_model().get_eos_token())],
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=False,  # Don't distribute validation
        )

        test_loader = self._merge_data_loaders(
            [dataset_class("test", EOS_TOKEN=self._get_model().get_eos_token())],
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=self.world_size > 1,
        )

        # Scheduler
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(WARMUP_FRAC * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        if self.rank == 0:
            print(f"📈 Total training steps: {total_steps}")
            print(f"🔥 Warmup steps: {warmup_steps}")

        # Load previous checkpoint if exists (for resuming current stage)
        best_epoch, best_val_loss = self._load_checkpoint(
            stage_name, optimizer, scheduler, eval_only=eval_only
        )
        if best_epoch is not None:
            print(
                f"📂 Resuming {stage_name} from epoch {best_epoch} (val_loss: {best_val_loss:.4f})"
            )
            # Display previous loss history if available
            self._display_loss_history(stage_name)
        else:
            print(f"🆕 Starting fresh training for {stage_name}")
            best_val_loss = float("inf")  # Ensure proper initialization

        # Skip training loop if eval_only is True
        if eval_only:
            if self.rank == 0:
                print(f"⏭️  Skipping training loop (eval_only mode)")
                print(f"📂 Using existing checkpoint for evaluation")
            epoch = best_epoch
            epochs_no_improve = 0
        else:
            # Training loop
            epochs_no_improve = 0
            start_epoch = best_epoch + 1 if best_epoch is not None else 1
            for epoch in range(start_epoch, num_epochs + 1):
                # Set epoch for distributed sampler
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

                # Training
                self.model.train()
                running_loss = 0.0
                prog = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch}/{num_epochs}",
                    disable=self.rank != 0,
                )
                for i, batch in enumerate(prog):
                    # DEBUG PRINT: Only for the first batch of the first epoch
                    if epoch == start_epoch and i == 0:
                        print(f"[DEBUG] Batch {i} - batch size: {len(batch)}")
                        if isinstance(batch, list) and isinstance(batch[0], dict):
                            for k, v in batch[0].items():
                                if hasattr(v, "shape"):
                                    print(f"[DEBUG] Sample key '{k}' shape: {v.shape}")
                                elif isinstance(v, list):
                                    print(
                                        f"[DEBUG] Sample key '{k}' list length: {len(v)}"
                                    )
                        import torch

                        print(
                            torch.cuda.memory_summary()
                            if torch.cuda.is_available()
                            else "No CUDA"
                        )
                    optimizer.zero_grad()
                    loss = self._get_model().compute_loss(batch)
                    loss.backward()

                    # Handle gradient clipping for distributed training
                    clip_grad_norm_(self._get_model().parameters(), GRAD_CLIP_NORM)

                    optimizer.step()
                    scheduler.step()

                    running_loss += loss.item()
                    if self.rank == 0:
                        prog.set_postfix(
                            loss=f"{loss.item():.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        )

                avg_train_loss = running_loss / len(train_loader)
                if self.rank == 0:
                    tqdm.write(f"Epoch {epoch} — train loss: {avg_train_loss:.4f}")

                # Validation
                val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm(
                        val_loader,
                        desc=f"Validating {stage_name}",
                        disable=self.rank != 0,
                    ):
                        val_loss += self._get_model().compute_loss(batch).item()

                avg_val_loss = val_loss / len(val_loader)

                # Synchronize validation loss across all ranks
                if dist.is_initialized():
                    val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    avg_val_loss = val_loss_tensor.item() / self.world_size

                if self.rank == 0:
                    tqdm.write(f"Epoch {epoch} — val   loss: {avg_val_loss:.4f}")
                    tqdm.write(f"Epoch {epoch} — best  loss: {best_val_loss:.4f}")

                # Save loss history for this epoch
                self._save_loss_history(stage_name, epoch, avg_train_loss, avg_val_loss)

                # Early stopping - all ranks need to make the same decision
                should_save = avg_val_loss + 1e-4 < best_val_loss
                if dist.is_initialized():
                    save_tensor = torch.tensor(
                        1 if should_save else 0, device=self.device
                    )
                    dist.all_reduce(save_tensor, op=dist.ReduceOp.SUM)
                    should_save = (
                        save_tensor.item() > 0
                    )  # If any rank thinks we should save, we save

                if should_save:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    self._save_checkpoint(
                        stage_name, epoch, avg_val_loss, optimizer, scheduler
                    )
                    if self.rank == 0:
                        tqdm.write("✔️  New best model saved.\n")
                else:
                    epochs_no_improve += 1
                    if self.rank == 0:
                        tqdm.write(
                            f"No improvement for {epochs_no_improve}/{EARLY_STOP_PAT} epochs.\n"
                        )

                    # Synchronize early stopping decision across all ranks
                    if epochs_no_improve >= EARLY_STOP_PAT:
                        if self.rank == 0:
                            tqdm.write(
                                f"\nEarly stopping triggered after {epoch} epochs."
                            )
                            tqdm.write(
                                f"Final stats: best_val_loss={best_val_loss:.4f}, epochs_no_improve={epochs_no_improve}"
                            )
                        break

                # Synchronize best_val_loss and epochs_no_improve across all ranks
                if dist.is_initialized():
                    best_loss_tensor = torch.tensor(best_val_loss, device=self.device)
                    epochs_tensor = torch.tensor(epochs_no_improve, device=self.device)
                    dist.broadcast(best_loss_tensor, src=0)
                    dist.broadcast(epochs_tensor, src=0)
                    best_val_loss = best_loss_tensor.item()
                    epochs_no_improve = int(epochs_tensor.item())

        # Load best model and evaluate
        best_epoch, _ = self._load_checkpoint(stage_name, optimizer, scheduler)
        if best_epoch is not None:
            if self.rank == 0:
                print(
                    f"📂 Loaded best checkpoint from epoch {best_epoch} for evaluation."
                )

        if self.rank == 0:
            if epoch is None:
                epoch = best_epoch
                print(f"🏁 Training completed for {stage_name}")
                print(f"   Total epochs run: {epoch}")
            else:
                print(f"🏁 Training completed for {stage_name}")
                print(f"   Total epochs run: {epoch}")
                print(f"   Best validation loss: {best_val_loss:.4f}")
                print(f"   Epochs without improvement: {epochs_no_improve}")

        metrics = self._evaluate_stage(
            stage_name, test_loader, stage_name, metric_func, best_epoch
        )

        return metrics

    def stage1_mcq(
        self, batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Stage 1: Multiple Choice Question Answering (TSQA).

        Configuration:
        - Epochs: 20
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Accuracy
        """
        return self._train_stage(
            stage_name="stage1_mcq",
            dataset_class=TSQADataset,
            num_epochs=30,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            eval_only=eval_only,
        )

    def stage2_captioning(
        self, batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Stage 2: Caption Generation (M4).

        Configuration:
        - Epochs: 15
        - OpenTSLMSP: encoder_lr=1e-4, projector_lr=5e-5 (lower for fine-tuning)
        - OpenTSLMFlamingo: base_lr=1e-4 (lower for fine-tuning)
        - Metric: Test loss only
        """
        return self._train_stage(
            stage_name="stage2_captioning",
            dataset_class=M4QADataset,
            num_epochs=20,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,  # Only test loss for captioning
            batch_size=batch_size,
            eval_only=eval_only,
        )

    def stage3_cot(
        self, batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Stage CoT: Chain-of-Thought Reasoning (HAR).

        Configuration:
        - Epochs: 100
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only (chain-of-thought reasoning)
        """
        sampler = None

        return self._train_stage(
            stage_name="stage3_cot",
            dataset_class=HARCoTQADataset,
            num_epochs=30,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,  # Only test loss for chain-of-thought reasoning
            batch_size=batch_size,
            eval_only=eval_only,
            sampler=sampler,
        )

    def stage4_sleep_cot(
        self, batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Stage 4: Chain-of-Thought Reasoning (SleepEDF).

        Configuration:
        - Epochs: 60
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only (chain-of-thought reasoning)
        """
        sampler = None

        return self._train_stage(
            stage_name="stage4_sleep_cot",
            dataset_class=SleepEDFCoTQADataset,
            num_epochs=60,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,  # Only test loss for chain-of-thought reasoning
            batch_size=batch_size,
            eval_only=eval_only,
            sampler=sampler,
        )

    def stage5_ecg_cot(
        self, batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Stage 5: Chain-of-Thought Reasoning (ECG QA CoT).

        Configuration:
        - Epochs: 60
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only (chain-of-thought reasoning)
        """
        sampler = None

        return self._train_stage(
            stage_name="stage5_ecg_cot",
            dataset_class=ECGQACoTQADataset,
            num_epochs=60,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,  # Only test loss for chain-of-thought reasoning
            batch_size=batch_size,
            eval_only=eval_only,
            sampler=sampler,
        )

    def stage4_sleep_cot(
        self, batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Stage 4: Chain-of-Thought Reasoning (SleepEDF).

        Configuration:
        - Epochs: 60
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only (chain-of-thought reasoning)
        """
        sampler = None

        return self._train_stage(
            stage_name="stage4_sleep_cot",
            dataset_class=SleepEDFCoTQADataset,
            num_epochs=60,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,  # Only test loss for chain-of-thought reasoning
            batch_size=batch_size,
            eval_only=eval_only,
            sampler=sampler,
        )

    def stage5_ecg_cot(
        self, batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Stage 5: Chain-of-Thought Reasoning (ECG QA CoT).

        Configuration:
        - Epochs: 60
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only (chain-of-thought reasoning)
        """
        sampler = None

        return self._train_stage(
            stage_name="stage5_ecg_cot",
            dataset_class=ECGQACoTQADataset,
            num_epochs=60,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,  # Only test loss for chain-of-thought reasoning
            batch_size=batch_size,
            eval_only=eval_only,
            sampler=sampler,
        )

    def stage6_ngafid_cot(
        self, batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Stage 6: Predictive Maintenance Reasoning (NGAFID CoT).

        Configuration:
        - Epochs: 60
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only (predictive maintenance reasoning)
        """
        sampler = None

        return self._train_stage(
            stage_name="stage6_ngafid_cot",
            dataset_class=NGAFIDCoTQADataset,
            num_epochs=30,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,  # Only test loss for predictive maintenance reasoning
            batch_size=batch_size,
            eval_only=eval_only,
            sampler=sampler,
        )

    def stage7_fault_detection_cot(
        self, batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Stage 7: Chain-of-Thought Reasoning (Fault Detection CoT).

        Configuration:
        - Epochs: 40
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only (chain-of-thought reasoning)
        """
        sampler = None

        return self._train_stage(
            stage_name="stage7_fault_detection_cot",
            dataset_class=FaultDetectionCoTQADataset,
            num_epochs=30,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size,
            eval_only=eval_only,
            sampler=sampler,
        )

    def run_curriculum(
        self, stages: List[str] = None, batch_size: int = None, eval_only: bool = False
    ):
        """Run the complete curriculum learning pipeline."""
        if stages is None:
            stages = CURRICULUM_STAGES

        # Filter out completed stages
        incomplete_stages = []
        for stage in stages:
            if self._is_stage_completed(stage):
                if self.rank == 0:
                    print(f"⏭️  Skipping completed stage: {stage}")
            else:
                incomplete_stages.append(stage)

        if self.rank == 0:
            print(f"🎓 Starting Curriculum Learning with {self.model_type}")
            if eval_only:
                print("🔍 EVAL-ONLY MODE: Will skip training and only run evaluation")
            print(f"📊 All stages: {', '.join(stages)}")
            print(f"🔄 Incomplete stages: {', '.join(incomplete_stages)}")
            print(f"💻 Device: {self.device}")
            if batch_size:
                print(f"📦 Batch size: {batch_size}")
            if self.world_size > 1:
                print(f"🌐 Distributed training with {self.world_size} GPUs")
            print("=" * 80)

        results = {}

        # Run only incomplete stages
        for stage in incomplete_stages:
            # Synchronize all ranks before starting each stage
            if dist.is_initialized():
                dist.barrier()

            if stage == "stage1_mcq":
                stage_results = self.stage1_mcq(
                    batch_size=batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
            elif stage == "stage2_captioning":
                stage_results = self.stage2_captioning(
                    batch_size=batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
            elif stage == "stage3_cot":
                stage_results = self.stage3_cot(
                    batch_size=batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
            elif stage == "stage4_sleep_cot":
                stage_results = self.stage4_sleep_cot(
                    batch_size=batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
            elif stage == "stage5_ecg_cot":
                stage_results = self.stage5_ecg_cot(
                    batch_size=batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
            elif stage == "stage6_ngafid_cot":
                stage_results = self.stage6_ngafid_cot(
                    batch_size=batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
            elif stage == "stage7_fault_detection_cot":
                stage_results = self.stage7_fault_detection_cot(
                    batch_size=batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
            elif stage == "stage4_sleep_cot":
                stage_results = self.stage4_sleep_cot(
                    batch_size=batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
            elif stage == "stage5_ecg_cot":
                stage_results = self.stage5_ecg_cot(
                    batch_size=batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
            else:
                if self.rank == 0:
                    print(f"⚠️  Unknown stage: {stage}, skipping...")

            # Synchronize all ranks after completing each stage
            if dist.is_initialized():
                dist.barrier()

        # Save overall results only on rank 0
        if self.rank == 0:
            overall_results_file = os.path.join(
                self.results_dir, "curriculum_results.json"
            )
            with open(overall_results_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\n🎉 Curriculum Learning Complete!")
            print(f"📁 All results saved to: {self.results_dir}/")
            print(f"📊 Overall results: {overall_results_file}")

        return results

    def _should_use_distributed(self) -> bool:
        """Check if distributed training should be used."""
        return ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1) or (
            "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) >= 0
        )

    def _init_distributed(self):
        """Initialize distributed training."""
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        elif "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])

        # Initialize process group
        dist.init_process_group(
            backend=self.dist_backend,
            init_method=self.dist_url,
            world_size=self.world_size,
            rank=self.rank,
            timeout=datetime.timedelta(hours=999),
        )

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)

        if self.rank == 0:
            print(f"Initialized distributed training with {self.world_size} GPUs")

    def _is_stage_completed(self, stage: str) -> bool:
        """Check if a stage is completed by verifying both training and evaluation were successful."""
        metrics_file = os.path.join(self.results_dir, stage, "results", "metrics.json")

        if not os.path.exists(metrics_file):
            return False

        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            # Check if the completion flag exists
            if not metrics.get("completed", False):
                return False

            # Check if evaluation was actually completed by looking for test_loss
            if "test_loss" not in metrics:
                return False

            # Check if test predictions file exists
            test_predictions_file = os.path.join(
                self.results_dir, stage, "results", "test_predictions.jsonl"
            )
            if not os.path.exists(test_predictions_file):
                return False

            return True

        except:
            return False

    def _mark_stage_completed(self, stage: str, metrics: Dict[str, Any]):
        """Mark a stage as completed by adding completion flag to metrics."""
        metrics["completed"] = True
        metrics["completion_epoch"] = metrics.get("epoch", "?")

        metrics_file = os.path.join(self.results_dir, stage, "results", "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        if self.rank == 0:
            print(f"✅ Stage {stage} marked as completed")

    def _get_model(self):
        """Get the underlying model (handles DDP wrapping)."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _checkpoint_exists(self, stage: str) -> bool:
        """Check if a checkpoint exists for a specific stage."""
        checkpoint_path = os.path.join(
            self.results_dir, stage, "checkpoints", "best_model.pt"
        )
        return os.path.exists(checkpoint_path)

    def _enable_lora_if_needed(self, stage_name: str):
        """Enable LoRA for OpenTSLMSP models in stages after stage2."""
        if self.model_type != "OpenTSLMSP":
            return  # LoRA only for OpenTSLMSP

        # Get the underlying model (handles DDP wrapping)
        model = self._get_model()

        # Enable LoRA for stages after stage2_captioning
        stages_with_lora = ["stage3_cot", "stage4_sleep_cot", "stage5_ecg_cot", "stage6_ngafid_cot"]

        if stage_name in stages_with_lora:
            if not getattr(model, "lora_enabled", False):
                if self.rank == 0:
                    print(f"🔧 Enabling LoRA for {stage_name}")
                try:
                    model.enable_lora(lora_r=16, lora_alpha=32, lora_dropout=0.0)
                    if self.rank == 0:
                        print(f"✅ LoRA enabled for {stage_name}")
                except Exception as e:
                    if self.rank == 0:
                        print(f"❌ Failed to enable LoRA for {stage_name}: {e}")
                        print("   Continuing without LoRA...")
            else:
                if self.rank == 0:
                    print(f"✅ LoRA already enabled for {stage_name}")
        else:
            if self.rank == 0:
                if stage_name in ["stage1_mcq", "stage2_captioning"]:
                    print(
                        f"ℹ️  LoRA disabled for {stage_name} (only enabled for stages 3+)"
                    )
                else:
                    print(f"ℹ️  LoRA not configured for {stage_name}")

    def _enable_lora_if_needed(self, stage_name: str):
        """Enable LoRA for OpenTSLMSP models in stages after stage2."""
        if self.model_type != "OpenTSLMSP":
            return  # LoRA only for OpenTSLMSP

        # Get the underlying model (handles DDP wrapping)
        model = self._get_model()

        # Enable LoRA for stages after stage2_captioning
        stages_with_lora = ["stage3_cot", "stage4_sleep_cot", "stage5_ecg_cot", "stage6_ngafid_cot"]

        if stage_name in stages_with_lora:
            if not getattr(model, "lora_enabled", False):
                if self.rank == 0:
                    print(f"🔧 Enabling LoRA for {stage_name}")
                try:
                    model.enable_lora(lora_r=16, lora_alpha=32, lora_dropout=0.0)
                    if self.rank == 0:
                        print(f"✅ LoRA enabled for {stage_name}")
                except Exception as e:
                    if self.rank == 0:
                        print(f"❌ Failed to enable LoRA for {stage_name}: {e}")
                        print("   Continuing without LoRA...")
            else:
                if self.rank == 0:
                    print(f"✅ LoRA already enabled for {stage_name}")
        else:
            if self.rank == 0:
                if stage_name in ["stage1_mcq", "stage2_captioning"]:
                    print(
                        f"ℹ️  LoRA disabled for {stage_name} (only enabled for stages 3+)"
                    )
                else:
                    print(f"ℹ️  LoRA not configured for {stage_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Learning for OpenTSLM Models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["OpenTSLMSP", "OpenTSLMFlamingo"],
        required=True,
        help="Model type to train",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=CURRICULUM_STAGES,
        default=CURRICULUM_STAGES,
        help="Stages to run (default: all stages)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training (default: use value from model_config.py)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval_only",
        default=False,
        action="store_true",
        help="Skip training and only run evaluation (requires existing checkpoint)",
    )

    # Model-specific arguments
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="LLM model ID for OpenTSLMFlamingo (e.g., 'google/medgemma-2b', 'meta-llama/Llama-3.2-1B')",
    )

    # Distributed training arguments
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="URL used to set up distributed training",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="Distributed backend"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local GPU rank",
    )

    # Logging arguments
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up global logging
    set_global_verbose(args.verbose)
    logger = get_logger(verbose=args.verbose)

    # Initialize trainer
    trainer = CurriculumTrainer(
        args.model,
        args.device,
        gradient_checkpointing=args.gradient_checkpointing,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend,
        local_rank=args.local_rank,
        llm_id=args.llm_id,
    )

    # Run curriculum
    results = trainer.run_curriculum(args.stages, args.batch_size, args.eval_only)

    # Print summary
    logger.info("Final Results Summary:")
    logger.info("=" * 40)
    for stage, metrics in results.items():
        logger.info(f"{stage.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
