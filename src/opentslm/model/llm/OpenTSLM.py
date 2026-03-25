# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
import torch
from typing import Optional, Union
from enum import Enum
from huggingface_hub import hf_hub_download

from .OpenTSLMSP import OpenTSLMSP
from .OpenTSLMFlamingo import OpenTSLMFlamingo


class ModelType(Enum):
    """Enumeration of supported model types."""

    SP = "sp"
    FLAMINGO = "flamingo"


class OpenTSLM:
    """
    Factory class for loading EmbedHealth models from Hugging Face Hub.

    Automatically detects model type based on repository ID suffix and returns
    the appropriate model instance (EmbedHealthSP or EmbedHealthFlamingo) with
    optimal parameters from curriculum learning training.

    - Repository IDs ending with "-sp" load EmbedHealthSP models
    - Repository IDs ending with "-flamingo" load EmbedHealthFlamingo models

    The factory automatically applies the exact same parameters used in curriculum learning:
    - EmbedHealthSP: Uses default constructor parameters
    - EmbedHealthFlamingo: cross_attn_every_n_layers=1, gradient_checkpointing=False

    These parameters are fixed and cannot be overridden since they were determined during training.

    Example:
        >>> model = OpenTSLM.load_pretrained("OpenTSLM/gemma-3-270m-pt-sleep-flamingo")
        >>>
        >>> from opentslm.prompt.full_prompt import FullPrompt
        >>> prompt = FullPrompt(...)
        >>> response = model.eval_prompt(prompt)
    """

    @classmethod
    def load_pretrained(
        cls,
        repo_id: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_lora: Optional[bool] = False,
    ) -> Union[OpenTSLMSP, OpenTSLMFlamingo]:
        """
        Load a pretrained model from Hugging Face Hub.

        Args:
            repo_id: Hugging Face repository ID (e.g., "OpenTSLM/gemma-3-270m-pt-sleep-flamingo")
            device: Device to load the model on (default: auto-detect)
            cache_dir: Directory to cache downloaded models (optional)
            enable_lora: Whether to enable LoRA (default: False)

        Returns:
            Union[OpenTSLMSP, OpenTSLMFlamingo]: The loaded model instance

        Example:
            >>> model = OpenTSLM.load_pretrained("OpenTSLM/gemma-3-270m-pt-sleep-flamingo")
            >>> prompt = FullPrompt(...)
            >>> response = model.eval_prompt(prompt)
        """
        device = cls._get_device(device)
        model_type = cls._detect_model_type(repo_id)
        checkpoint_path = cls._download_model_files(repo_id, cache_dir)
        base_llm_id = cls._get_base_llm_id(repo_id)

        print(f"🚀 Loading {model_type.value.upper()} model...")
        print(f"   Repository: {repo_id}")
        print(f"   Base LLM: {base_llm_id}")
        print(f"   Device: {device}")

        # Instantiate model with fixed training parameters
        if model_type == ModelType.SP:
            # OpenTSLMSP uses default parameters from curriculum learning
            model = OpenTSLMSP(llm_id=base_llm_id, device=device)
            if enable_lora:
                model.enable_lora()
        elif model_type == ModelType.FLAMINGO:
            # OpenTSLMFlamingo with fixed parameters from curriculum learning
            model = OpenTSLMFlamingo(
                device=device,
                llm_id=base_llm_id,
                cross_attn_every_n_layers=1,
                gradient_checkpointing=False,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load the checkpoint
        model.load_from_file(checkpoint_path)
        model.eval()

        print(f"✅ {model_type.value.upper()} model loaded successfully!")
        return model

    @staticmethod
    def _get_device(device: Optional[str]) -> str:
        """Auto-detect device if not specified."""
        if device is not None:
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def _detect_model_type(repo_id: str) -> ModelType:
        """Detect model type from repository ID suffix."""
        if repo_id.endswith("-sp"):
            return ModelType.SP
        elif repo_id.endswith("-flamingo"):
            return ModelType.FLAMINGO
        else:
            raise ValueError(
                f"Repository ID '{repo_id}' must end with either '-sp' or '-flamingo' "
                f"to indicate the model type."
            )

    @staticmethod
    def _download_model_files(repo_id: str, cache_dir: Optional[str] = None) -> str:
        """Download model checkpoint from Hugging Face Hub."""
        try:
            # Download the main model checkpoint file
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename="model_checkpoint.pt",
                cache_dir=cache_dir,
                local_files_only=False,
            )
            print(f"✅ Downloaded model checkpoint from {repo_id}")
            return checkpoint_path

        except Exception as e:
            raise RuntimeError(
                f"Failed to download model from {repo_id}. "
                f"Tried 'model_checkpoint.pt'. "
                f"Original error: {e}"
            )

    @staticmethod
    def _get_base_llm_id(repo_id: str) -> str:
        """Get the base LLM ID from static mapping based on repository ID pattern."""
        repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # Extract base model from repository name pattern
        if repo_name.startswith("llama-3.2-3b"):
            return "meta-llama/Llama-3.2-3B"
        elif repo_name.startswith("llama-3.2-1b"):
            return "meta-llama/Llama-3.2-1B"
        elif repo_name.startswith("gemma-3-1b"):
            return "google/gemma-3-1b"
        elif repo_name.startswith("gemma-3-270m"):
            return "google/gemma-3-270m"
        else:
            # Raise exception if pattern doesn't match
            raise ValueError(
                f"Unable to determine base LLM ID from repository name '{repo_name}'. "
                f"Repository name must start with one of: 'llama-3.2-3b', 'llama-3.2-1b', "
                f"'gemma-3-1b', or 'gemma-3-270m'."
            )
