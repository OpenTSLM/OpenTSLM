# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

__all__ = ["OpenTSLM"]


def __getattr__(name: str):
    if name == "OpenTSLM":
        from opentslm.model.llm.OpenTSLM import OpenTSLM

        return OpenTSLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")