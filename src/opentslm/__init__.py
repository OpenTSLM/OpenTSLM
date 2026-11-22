# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentslm.model.llm.OpenTSLM import OpenTSLM

__all__ = ["OpenTSLM"]


def __getattr__(name: str) -> Any:
    if name == "OpenTSLM":
        # Defer importing LLM backends when only standalone encoder modules are used.
        from opentslm.model.llm.OpenTSLM import OpenTSLM

        globals()[name] = OpenTSLM
        return OpenTSLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
