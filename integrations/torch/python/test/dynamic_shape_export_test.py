# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import pytest
import torch
from iree.importers.torch.exporter import export


@pytest.mark.parametrize(
    "import_symbolic_shape_expressions",
    [
        True,
        False,
    ],
)
def test_exported_program_dynamic_shapes(import_symbolic_shape_expressions):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.branch1 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU())
            self.branch2 = torch.nn.Sequential(
                torch.nn.Linear(128, 64), torch.nn.ReLU()
            )
            self.buffer = torch.ones(32)

        def forward(self, x1, x2):
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
            return (out1 + self.buffer, out2)

    example_args = (torch.randn(32, 64), torch.randn(32, 128))

    # Create a dynamic batch size
    batch = torch.export.Dim("batch")
    # Specify that the first dimension of each input is that batch size
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    output = export(
        M(),
        args=example_args,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=import_symbolic_shape_expressions,
    )
    output.print_readable()
    asm = str(output.mlir_module)

    if import_symbolic_shape_expressions:
        assert "bind_symbolic_shape" in asm
    else:
        assert "bind_symbolic_shape" not in asm


if __name__ == "__main__":
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "-vv"] + args)

    sys.exit(ret_code)
