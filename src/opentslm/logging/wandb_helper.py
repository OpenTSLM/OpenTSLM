from typing import Any, Dict, List, Optional

import datetime
import wandb
import torch


class WandbHelper:
    """Minimal wandb wrapper that owns run initialization."""

    def __init__(
        self,
        *,
        model_type: str,
        llm_id: Optional[str],
        llm_id_safe: str,
        encoder_type: Optional[str],
        device: str,
        world_size: int,
        rank: int,
        gradient_checkpointing: bool,
        wandb_project: str,
        wandb_entity: Optional[str]=None,
        wandb_run_name: Optional[str]=None,
        wandb_tags: Optional[List[str]]=[],
        disabled: bool = False,
    ) -> None:
        self.disabled = disabled
        self.run = None
        self.initialized = False
        self.model_type = model_type
        self.llm_id = llm_id
        self.llm_id_safe = llm_id_safe
        self.encoder_type = encoder_type
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.gradient_checkpointing = gradient_checkpointing
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_tags = wandb_tags or []

    def init_wandb(self, stage_name: Optional[str] = None, resume: bool = False) -> Optional[Any]:
        """Initialize wandb for tracking experiments."""
        if self.disabled:
            return None
        try:
            # Build a fresh run name per stage without mutating the base name
            base_run_name = self.wandb_run_name
            if not base_run_name:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_run_name = f"{self.model_type}_{self.llm_id_safe}_{timestamp}"

            stage_run_name = (
                f"{base_run_name}_{stage_name}" if stage_name else base_run_name
            )

            # Prepare tags
            tags = list(self.wandb_tags)
            tags.extend([self.model_type, self.llm_id_safe])
            if self.model_type == "OpenTSLMFlamingo":
                tags.append(f"enc_{self.encoder_type}")
            if stage_name:
                tags.append(stage_name)
            if self.world_size > 1:
                tags.append("distributed")

            # Prepare config
            wandb_config = {
                "model_type": self.model_type,
                "llm_id": self.llm_id,
                "device": self.device,
                "world_size": self.world_size,
                "rank": self.rank,
                "gradient_checkpointing": self.gradient_checkpointing,
                "stage": stage_name,
            }
            if self.model_type == "OpenTSLMFlamingo":
                wandb_config["encoder_type"] = self.encoder_type

            self.run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=stage_run_name,
                tags=tags,
                resume=resume,
                config=wandb_config,
            )
            self.initialized = True

            if self.rank == 0 and self.run is not None and hasattr(self.run, "url"):
                print(f"🔬 Wandb initialized: {self.run.url}")

            return self.run
        except Exception as exc:
            if self.rank == 0:
                print(f"⚠️  Failed to initialize wandb: {exc}")
                print("   Continuing without wandb logging...")
            self.run = None
            self.initialized = False
            self.disabled = True
            return None

    def log_metrics(
        self,
        *,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """Log metrics to wandb."""
        if not self.initialized or self.disabled or self.run is None:
            return
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.run.log(metrics, step=step)

    def log_model_info(self, model: Any) -> None:
        """Log model architecture and system information to wandb."""
        if not self.initialized or self.disabled or self.run is None:
            return
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Log model information
        model_info = {
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/parameter_ratio": trainable_params / total_params
            if total_params > 0
            else 0,
            "model/model_type": self.model_type,
            "model/llm_id": self.llm_id,
            "system/device": self.device,
            "system/world_size": self.world_size,
            "system/rank": self.rank,
        }

        # Add GPU memory info if available
        if torch.cuda.is_available():
            model_info.update(
                {
                    "system/gpu_memory_allocated": torch.cuda.memory_allocated()
                    / 1024**3,  # GB
                    "system/gpu_memory_reserved": torch.cuda.memory_reserved()
                    / 1024**3,  # GB
                    "system/gpu_count": torch.cuda.device_count(),
                }
            )

        self.run.log(model_info, step=0)

    def update_config(self, config: Dict[str, Any]) -> None:
        if not self.initialized or self.disabled or self.run is None:
            return
        self.run.config.update(config)

    def finish(self) -> None:
        if self.initialized and not self.disabled and self.run is not None:
            self.run.finish()
            self.initialized = False
            self.run = None
