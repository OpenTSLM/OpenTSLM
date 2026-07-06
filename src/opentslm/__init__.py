# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import os

# Runtime type/shape checking is opt-in via OPENTSLM_MODE, evaluated once at import.
#   "opt" (default): jaxtyping annotations stay inert -> zero overhead, and
#                    beartype is not required to import the package.
#   "dev":           install jaxtyping's beartype import hook so that Float[...]
#                    shape annotations (and ordinary type hints) across the whole
#                    opentslm package are enforced at runtime.
# The hook must be installed before any opentslm submodule is imported, so this
# block stays at the top of the package __init__, ahead of the imports below.
if os.environ.get("OPENTSLM_MODE", "opt").lower() == "dev":
    from jaxtyping import install_import_hook

    install_import_hook("opentslm", "beartype.beartype")

from opentslm.model.llm.OpenTSLM import OpenTSLM

__all__ = ["OpenTSLM"]