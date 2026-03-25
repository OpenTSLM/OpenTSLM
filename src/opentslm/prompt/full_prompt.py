# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from typing import List
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt


class FullPrompt:
    """
    Combines multiple TextTimeSeriesPrompts with a pre- and post-prompt.
    """

    def __init__(
        self,
        pre_prompt: TextPrompt,
        text_time_series_prompt_list: List[TextTimeSeriesPrompt],
        post_prompt: TextPrompt,
    ):
        assert isinstance(pre_prompt, TextPrompt), "Pre prompt must be a TextPrompt."
        assert isinstance(post_prompt, TextPrompt), "Post prompt must be a TextPrompt."

        self.pre_prompt = pre_prompt
        self.text_time_series_prompt_texts = list(
            map(lambda x: x.get_text(), text_time_series_prompt_list)
        )
        self.text_time_series_prompt_time_series = list(
            map(lambda x: x.get_time_series(), text_time_series_prompt_list)
        )
        self.post_prompt = post_prompt

    def to_dict(self):
        return {
            "post_prompt": self.post_prompt.get_text(),
            "pre_prompt": self.pre_prompt.get_text(),
            "time_series": self.text_time_series_prompt_time_series,
            "time_series_text": self.text_time_series_prompt_texts,
        }
