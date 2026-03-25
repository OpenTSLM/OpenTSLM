# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from .prompt import Prompt


class TextPrompt(Prompt):
    def __init__(self, text: str):
        assert isinstance(text, str), "Text must be a string!"
        self.__text = text

    def get_text(self) -> str:
        return self.__text
