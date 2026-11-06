# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
import sys
import textwrap
import unittest


def _run_alone(script):
    """`script` in a fresh process, without the key preamble."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
    )


class TorchaxImportTest(unittest.TestCase):
    def test_import_loads_neither_torchax_nor_jax(self):
        result = _run_alone(
            """
            import sys
            import torch

            backend = torch._C._get_privateuse1_backend_name()
            import iree_torchax

            assert "torchax" not in sys.modules
            assert "jax" not in sys.modules
            assert torch._C._get_privateuse1_backend_name() == backend

            class Affine(torch.nn.Module):
                def forward(self, x):
                    return x * 2 + 1

            torch.export.export(Affine(), (torch.ones(3),), strict=True)
            iree_torchax.exported_program_to_jax
            assert "torchax" in sys.modules
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_child_export_preserves_parent_torch_state(self):
        result = _run_alone(
            """
            import pathlib
            import subprocess
            import sys
            import tempfile
            import torch

            class Affine(torch.nn.Module):
                def forward(self, x):
                    return x * 2 + 1

            backend = torch._C._get_privateuse1_backend_name()
            with tempfile.TemporaryDirectory() as directory:
                output = pathlib.Path(directory) / "model.mlir"
                child = (
                    "import pathlib, sys, torch, iree_torchax; "
                    "Affine = type('Affine', (torch.nn.Module,), "
                    "{'forward': lambda self, x: x * 2 + 1}); "
                    "exported = torch.export.export(Affine(), (torch.ones(3),)); "
                    "_, artifact = iree_torchax.exported_program_to_stablehlo(exported); "
                    "pathlib.Path(sys.argv[1]).write_text(str(artifact.mlir_module()))"
                )
                subprocess.run([sys.executable, "-c", child, str(output)], check=True)
                assert 'func.func public @main' in output.read_text()

            assert "torchax" not in sys.modules
            assert "jax" not in sys.modules
            assert torch._C._get_privateuse1_backend_name() == backend
            torch.export.export(Affine(), (torch.ones(3),), strict=True)
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
