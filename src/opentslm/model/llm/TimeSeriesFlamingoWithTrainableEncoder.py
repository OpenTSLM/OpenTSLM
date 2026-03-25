# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
from torch import nn
from open_flamingo import Flamingo
from einops import rearrange


class TimeSeriesFlamingoWithTrainableEncoder(Flamingo):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(vision_encoder, lang_encoder, eoc_token_id, media_token_id, vis_dim, cross_attn_every_n_layers, gradient_checkpointing)

    # Override the _encode_vision_x method to handle time series data
    # In the original Flamingo, the vision_encoder is a CLIPModel, which is not trainable (with torch.no_grad())
    # Here, we use a TimeSeriesCNNEncoder, which is trainable
    def _encode_vision_x(self, vision_x):
        # Handle time series data while still using the TimeSeriesCNNEncoder
        if vision_x.ndim == 4:  # For shape (b, T_img, F, features)
            b, T, F, features = vision_x.shape
            
            # Flatten batch, time and frame dimensions
            vision_x = rearrange(vision_x, "b T F c -> (b T F) c")
            
            # Process through encoder - will return [batch, patches, features]
                
            vision_x = self.vision_encoder(vision_x)  # Shape: [(b*T*F), patches, features]
                
            # Reshape to expected format for perceiver
            # The transformer output already has the "tokens" dimension we need (patches)
            vision_x = rearrange(vision_x, "(b T F) p d -> b T F p d", b=b, T=T, F=F)
            
            # Process through perceiver
            vision_x = self.perceiver(vision_x)
            
        else:
            # Original image processing path
            assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
            b, T, F = vision_x.shape[:3]
            assert F == 1, "Only single frame supported"

            vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
            
            vision_x = self.vision_encoder(vision_x)[1]
            vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
            vision_x = self.perceiver(vision_x)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)
