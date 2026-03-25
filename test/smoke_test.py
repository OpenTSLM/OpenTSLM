# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import importlib

mods = [
    "opentslm",
    "opentslm.model.llm.OpenTSLM",
    "opentslm.model.llm.OpenTSLMFlamingo",
    "opentslm.prompt.full_prompt",
]

[importlib.import_module(m) for m in mods]
print("Smoke test passed: all modules imported successfully.")
