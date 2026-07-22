# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Self-contained check that the jaxtyping shape annotations in the codebase are
actually enforced at runtime.

jaxtyping annotations such as ``Float[torch.Tensor, "*batch patches encoder_dim"]``
are inert on their own -- Python never validates annotations at call time. They
only become runtime shape checks when combined with a runtime typechecker
(beartype here) via jaxtyping's import hook. This test installs that hook for a
single module and exercises one annotated function with a correct and an
incorrect shape.

``LinearProjector.forward`` is a good probe because ``torch.nn.Linear`` happily
accepts a 1-D input (it just treats the last axis as features), so a 1-D tensor
is rejected *only* by the jaxtyping annotation -- if the bad-shape call raises,
the annotation is provably being enforced.
"""

import sys
import unittest

import torch
from jaxtyping import TypeCheckError, install_import_hook

TARGET = "opentslm.model.projector.LinearProjector"

# The import hook can only instrument a module imported *after* it is installed,
# so drop any previously-cached copy to guarantee a fresh, instrumented import.
for _name in list(sys.modules):
    if _name == TARGET or _name.startswith(TARGET + "."):
        del sys.modules[_name]

with install_import_hook(TARGET, "beartype.beartype"):
    from opentslm.model.projector.LinearProjector import LinearProjector


class JaxtypingShapeCheckTest(unittest.TestCase):
    """Verifies jaxtyping annotations are enforced at runtime via beartype."""

    def setUp(self):
        self.projector = LinearProjector(input_dim=8, output_dim=16, device="cpu")

    def test_correct_shapes_pass(self):
        """A well-shaped [*batch, patches, encoder_dim] tensor passes the check."""
        x = torch.randn(2, 3, 8)  # batch=2, patches=3, encoder_dim=8
        out = self.projector(x)
        self.assertEqual(out.shape, (2, 3, 16))

    def test_incorrect_shape_raises(self):
        """A 1-D tensor lacks the required patches/encoder_dim axes and must be rejected.

        torch.nn.Linear(8, 16) would silently accept this 1-D tensor and return
        shape [16], so a raised error proves the jaxtyping annotation is enforced
        rather than torch doing the checking.
        """
        bad = torch.randn(8)
        with self.assertRaises(
            TypeCheckError,
            msg="LinearProjector.forward accepted a 1-D tensor; is the beartype "
            "import hook installed correctly?",
        ):
            self.projector(bad)


if __name__ == "__main__":
    unittest.main()
