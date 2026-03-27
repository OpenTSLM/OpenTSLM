from threading import Thread
from typing import Any, Callable, Dict, Iterator, List

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

    def stream_generate(
        self, batch: List[Dict[str, Any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> Iterator[str]:
        raise NotImplementedError(
            "stream_generate method should be implemented by the subclass"
        )

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

    def stream_prompt(
        self, prompt: FullPrompt, max_new_tokens: int = 1000, normalize: bool = False
    ) -> Iterator[str]:
        raise NotImplementedError(
            "stream_prompt method should be implemented by the subclass"
        )

    @staticmethod
    def _validate_streaming_batch(batch: List[Dict[str, Any]]) -> None:
        if len(batch) != 1:
            raise ValueError(
                "Streaming generation currently supports exactly one sample per batch."
            )

    @staticmethod
    def _iterate_streamer(streamer: Any, generate_fn: Callable[[], None]) -> Iterator[str]:
        error: BaseException | None = None

        def runner() -> None:
            nonlocal error
            try:
                generate_fn()
            except BaseException as exc:  # pragma: no cover - re-raised in caller
                error = exc
                end = getattr(streamer, "end", None)
                if callable(end):
                    end()

        thread = Thread(target=runner, daemon=True)
        thread.start()

        try:
            for text in streamer:
                if text:
                    yield text
        finally:
            thread.join()

        if error is not None:
            raise error
