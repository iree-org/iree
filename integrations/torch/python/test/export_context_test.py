# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import pytest
import torch
import torch.nn as nn
from iree.importers.torch.export_context import ExportContext
from iree.importers.torch.exporter import ModuleBuilderOptions


class SimpleParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(20, 30)

    def forward(self, x):
        return self.classifier(x)


def test_export_basic():
    m = SimpleParams()
    ctx = ExportContext(
        module_name="main",
        options=ModuleBuilderOptions(import_symbolic_shape_expressions=True),
    )
    torch_exported = torch.export.export(m, (torch.randn(20, 20),))
    ctx.add_exported_program("test", torch_exported, True)
    print("hello")
    print(ctx.get_mlir_module())


if __name__ == "__main__":
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "-vv"] + args)

    sys.exit(ret_code)
