# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from .prompt import Prompt
import numpy as np
from collections.abc import Sequence


class TextTimeSeriesPrompt(Prompt):
    """
    A prompt that includes text and an associated time series.

    Attributes:
        text (str): The input text.
        time_series (np.ndarray): The associated time series data.
    """

    def __init__(self, text: str, time_series: Sequence):
        assert isinstance(text, str), "Text must be a string!"
        assert isinstance(time_series, (np.ndarray, Sequence)), (
            "Time series must be a list or numpy array!"
        )

        ts_array = np.array(time_series)

        assert ts_array.ndim == 1, "Time series must be one-dimensional! You can input multiple time series, but each time series must be a one-dimensional array."
        assert ts_array.size > 0, "Time series must not be empty!"
        assert np.issubdtype(ts_array.dtype, np.number), (
            "Time series must contain only numeric values!"
        )

        super().__init__()

        self.__text = text
        self.__time_series = ts_array

    def get_text(self) -> str:
        return self.__text

    def get_time_series(self) -> np.ndarray:
        return self.__time_series
