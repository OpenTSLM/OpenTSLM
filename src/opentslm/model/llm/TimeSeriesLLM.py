from typing import List, Dict, Any

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from opentslm.prompt.full_prompt import FullPrompt

class TimeSeriesLLM(nn.Module):
    def __init__(
        self,
        device,
    ):
        super().__init__()
        self.device = device

    
    def generate(
        self, batch: List[Dict[str, Any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        
        raise NotImplementedError("Generate method should be implemented by the subclass")

    def compute_loss(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """
        batch: same format as generate()
        answers: List[str] of length B
        """
        raise NotImplementedError("Compute loss method should be implemented by the subclass")

    def get_eos_token(self) -> str:
        raise NotImplementedError("Get eos token method should be implemented by the subclass")

    def eval_prompt(self, prompt: FullPrompt) -> str:
        raise NotImplementedError("Eval prompt method should be implemented by the subclass")