# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import sys
import unittest

import pytest
import torch
import torch.nn as nn
from iree.importers.torch.exporter import export
from iree.importers.torch.params import externalize_module_parameters


class ExportAPI(unittest.TestCase):
    def testStaticNNModule(self):
        mdl = SimpleParams()
        exported = export(mdl, args=(torch.empty([128, 20]),))
        exported.print_readable()
        asm = str(exported.mlir_module)
        self.assertIn("dense_resource", asm)

    def testDynamicNNModule(self):
        mdl = SimpleParams()
        batch = torch.export.Dim("batch")
        exported = export(
            mdl, args=(torch.empty([128, 20]),), dynamic_shapes={"x": {0: batch}}
        )
        exported.print_readable()
        asm = str(exported.mlir_module)
        self.assertIn(
            "func.func @main(%arg0: !torch.vtensor<[?,20],f32>) -> !torch.vtensor<[?,30],f32>",
            asm,
        )

    def testExternalParamsNNModule(self):
        mdl = SimpleParams()
        externalize_module_parameters(mdl)
        exported = export(mdl, args=(torch.empty([128, 20]),))
        exported.print_readable()
        asm = str(exported.mlir_module)
        self.assertNotIn("dense_resource", asm)
        self.assertIn("util.global.load", asm)

    def testTorchExportedProgram(self):
        mdl = SimpleParams()
        externalize_module_parameters(mdl)
        prg = torch.export.export(mdl, args=(torch.empty([128, 20]),))
        exported = export(prg)
        exported.print_readable()
        asm = str(exported.mlir_module)
        self.assertNotIn("dense_resource", asm)
        self.assertIn(
            'util.global private @__auto.classifier.weight = #flow.parameter.named<"model"::"classifier.weight">',
            asm,
        )
        self.assertIn(
            'util.global private @__auto.classifier.bias = #flow.parameter.named<"model"::"classifier.bias">',
            asm,
        )

    def testUnsupportedExportedProgram(self):
        class UnsupportedExportType: ...

        with self.assertRaises(TypeError):
            export(UnsupportedExportType)


class SimpleParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(20, 30)

    def forward(self, x):
        return self.classifier(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "-vv"] + args)

    sys.exit(ret_code)
