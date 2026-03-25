# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Literal, Tuple

import numpy as np
from opentslm.prompt.prompt_with_answer import PromptWithAnswer
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from torch.utils.data import Dataset


class QADataset(Dataset, ABC):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
    ):
        """
        Initializes the dataset by loading and formatting the specified data split.
        Args:
            split (Literal["train", "test", "validation"]): The dataset split to load. Must be one of "train", "test", or "validation".
            EOS_TOKEN (str): End-of-sequence token to be used in formatting.
            format_sample_str (bool, optional): If True, applies a string formatting function to each sample. Defaults to False.
            time_series_format_function (Callable[[np.ndarray], str] | None, optional): Optional function to format time series data as strings. Used only if `format_sample_str` is True.
        Raises:
            RuntimeError: If the provided split is not one of "train", "test", or "validation".
        Notes:
            - The datasets for each split are loaded and formatted only once per class.
            - The formatted datasets are cached as class attributes for subsequent initializations.
        """
        
        self.EOS_TOKEN = EOS_TOKEN
        if not hasattr(self.__class__, "loaded"):
            train, val, test = self._load_splits()

            format_function = partial(self._format_sample_str, time_series_format_function) if format_sample_str else self._format_sample
           
            from tqdm import tqdm
            
            print("Formatting training samples...")
            self.__class__._train_dataset = list(tqdm(map(format_function, train), total=len(train), desc="Training samples"))
            
            print("Formatting validation samples...")
            self.__class__._validation_dataset = list(tqdm(map(format_function, val), total=len(val), desc="Validation samples"))
            
            print("Formatting test samples...")
            self.__class__._test_dataset = list(tqdm(map(format_function, test), total=len(test), desc="Test samples"))

            self.__class__.loaded = True

        match split:
            case "train":
                self.dataset = self.__class__._train_dataset
            case "validation":
                self.dataset = self.__class__._validation_dataset
            case "test":
                self.dataset = self.__class__._test_dataset
            case _:
                raise RuntimeError(
                    "Split must be a literal of 'train', 'training', or 'validation'"
                )

    @abstractmethod
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        pass

    @abstractmethod
    def _get_answer(self, row) -> str:
        pass

    @abstractmethod
    def _get_pre_prompt(self, row) -> str:
        pass

    @abstractmethod
    def _get_post_prompt(self, row) -> str:
        pass

    @abstractmethod
    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        pass

    def _format_sample(self, row):
        answer = self._get_answer(row)
        if not answer.endswith(self.EOS_TOKEN):
            answer += self.EOS_TOKEN

        return PromptWithAnswer(
            TextPrompt(self._get_pre_prompt(row).strip()),
            self._get_text_time_series_prompt_list(row),
            TextPrompt(self._get_post_prompt(row).strip()),
            answer.strip(),
        ).to_dict()

    def _format_sample_str(
        self, time_series_format_function: Callable[[np.ndarray], str] | None, row
    ):
        def fallback_timeseries_formatter(time_series: np.ndarray) -> str:
            # Fallback formatter for time series data
        
            return np.array2string(
                time_series,
                separator=" ",
                formatter={"all": lambda x: f'"{x:.2f}"'.replace(".", "")},
                threshold=sys.maxsize,
                max_line_width=sys.maxsize,
            ).removeprefix("[").removesuffix("]")

               

        if not time_series_format_function:
            time_series_format_function = fallback_timeseries_formatter

        # Create the prompt chunks: pre-prompt, time series prompts, and post-prompt
        prompt_chunks = [self._get_pre_prompt(row).strip()]

        for text_time_series_prompt in self._get_text_time_series_prompt_list(row):
            prompt_chunks.append(text_time_series_prompt.get_text())
            time_series = time_series_format_function(
                text_time_series_prompt.get_time_series()
            )
            prompt_chunks.append(time_series)

        prompt_chunks.append(self._get_post_prompt(row).strip())

        return {"prompt": "\n".join(prompt_chunks), "answer": self._get_answer(row)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
