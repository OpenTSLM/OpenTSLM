#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
from typing import Optional

from model_config import ENCODER_OUTPUT_DIM
from model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase

try:
    from chronos import Chronos2Model
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    Chronos2Model = None


class Chronos2Encoder(TimeSeriesEncoderBase):
    """
    Chronos-2 encoder wrapper for OpenTSLMFlamingo.
    
    This encoder uses the pre-trained Chronos-2 model's encoder to extract
    features from time series data. The encoder outputs are projected to
    match the expected output dimension.
    
    Args:
        model_name: HuggingFace model name for Chronos-2 (default: "amazon/chronos-2")
        output_dim: Output dimension (default: ENCODER_OUTPUT_DIM=128)
        dropout: Dropout probability (default: 0.0)
        freeze_backbone: Whether to freeze the Chronos-2 backbone (default: False)
        device: Device to load the model on (default: None, uses input device)
    """
    
    def __init__(
        self,
        model_name: str = "amazon/chronos-2",
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        device: Optional[str] = None,
        use_group_ids: bool = False,
    ):
        super().__init__(output_dim, dropout)
        
        if not CHRONOS_AVAILABLE:
            raise ImportError(
                "chronos-forecasting package is not installed. "
                "Please install it with: pip install 'chronos-forecasting>=2.0'"
            )
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_group_ids = use_group_ids
        
        # Load Chronos-2 model
        # Note: We only need the encoder part, but we load the full model
        # and access its encoder component
        print(f"Loading Chronos-2 model from {model_name}...")
        self.chronos_model = Chronos2Model.from_pretrained(model_name)
        
        if device is not None:
            self.chronos_model = self.chronos_model.to(device)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.chronos_model.parameters():
                param.requires_grad = False
            print("Chronos-2 backbone frozen")
        else:
            # Only freeze the output layers, keep encoder trainable
            # The encoder will be fine-tuned
            if hasattr(self.chronos_model, 'output_patch_embedding'):
                for param in self.chronos_model.output_patch_embedding.parameters():
                    param.requires_grad = False
        
        # Get the encoder's hidden dimension
        # Chronos-2 uses d_model=768 for the encoder
        encoder_hidden_dim = 768  # This is Chronos-2's d_model
        
        # Create projection layer to map from encoder hidden dim to output dim
        self.projection = nn.Linear(encoder_hidden_dim, output_dim)

        # Normalize projected tokens to keep scale consistent with CNNTokenizer
        self.output_norm = nn.LayerNorm(output_dim)

        # Dropout layer
        self.output_dropout = nn.Dropout(self.dropout)
        
        print(f"Chronos2Encoder initialized: {encoder_hidden_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Chronos-2 encoder.
        
        Args:
            x: FloatTensor of shape [B, L], a batch of raw time series.
               Where B is batch size and L is sequence length.
        
        Returns:
            FloatTensor of shape [B, N, output_dim], where N is the number
            of patches/tokens after encoding.
        """
        B, L = x.shape
        
        # Move input to the same device as the model
        device = next(self.chronos_model.parameters()).device
        x = x.to(device)
        
        # Forward through Chronos-2 model to get encoder outputs
        # We use a hook to extract the encoder's hidden states
        # This is necessary because Chronos2Model doesn't expose encoder outputs directly
        encoder_outputs = []
        
        def hook_fn(module, input, output):
            # Extract last_hidden_state from Chronos2EncoderOutput
            if hasattr(output, 'last_hidden_state'):
                encoder_outputs.append(output.last_hidden_state)
            elif isinstance(output, tuple) and len(output) > 0:
                encoder_outputs.append(output[0])
        
        # Register hook on encoder
        handle = self.chronos_model.encoder.register_forward_hook(hook_fn)
        
        try:
            # Forward through the model
            # We call the full model forward, but only use the encoder outputs
            # num_output_patches=1 minimizes computation since we don't need predictions
            with torch.set_grad_enabled(not self.freeze_backbone):
                forward_kwargs = dict(
                    context=x,
                    num_output_patches=1,  # Minimal output patches since we only need encoder
                    output_attentions=False,
                )
                # For univariate time series, Chronos2 model handles grouping internally.
                # Passing group_ids=None follows Chronos2 model.py recommendations.
                if self.use_group_ids:
                    forward_kwargs["group_ids"] = torch.zeros(
                        B, dtype=torch.long, device=device
                    )
                _ = self.chronos_model(**forward_kwargs)
        finally:
            handle.remove()
        
        # Extract encoder hidden states
        if not encoder_outputs:
            raise RuntimeError("Failed to extract encoder outputs from Chronos-2 model")
        
        hidden_states = encoder_outputs[0]  # Shape: [B, N, 768]
        
        # Project to output dimension
        # hidden_states: [B, N, 768] -> [B, N, output_dim]
        projected = self.projection(hidden_states)

        # Normalize + dropout (matches CNNTokenizer behavior which uses LayerNorm)
        projected = self.output_norm(projected)
        output = self.output_dropout(projected)
        
        return output
    
    def get_device(self):
        """Get the device of the model."""
        return next(self.chronos_model.parameters()).device

