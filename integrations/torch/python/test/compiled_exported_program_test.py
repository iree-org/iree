# Copyright 2024 Advanced Micro Devices, Inc.
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
from iree.importers.torch.fx_programs import FxProgramsBuilder
from iree.importers.torch.params import externalize_module_parameters


class TorchExportTests(unittest.TestCase):
    def testMultiPublic(self):
        class MyModule(torch.nn.Module):
            def forward(self): ...

        fxb = FxProgramsBuilder(MyModule())

        @fxb.export_program(
            args=([torch.empty([3, 2]), torch.empty([1, 2])],),
            kwargs={"foobar": torch.empty([3, 1])},
        )
        def _compute1(module, inputs, *, foobar):
            t1 = inputs[0]
            t2 = inputs[1]
            t3 = t1 + t2 + foobar
            return [t3 * t3, foobar]

        @fxb.export_program(
            args=([torch.empty([5]), torch.empty([5])],),
            kwargs={"foobar": torch.empty([5])},
        )
        def _compute2(module, inputs, *, foobar):
            t1 = inputs[0]
            t2 = inputs[1]
            t3 = t1 + t2 + foobar
            return [t3 * t3, foobar]

        exported = export(fxb)
        module_str = str(exported.mlir_module)
        self.assertIn("func.func @_compute1", module_str)
        self.assertIn("func.func @_compute2", module_str)

    def testParametersAsGlobalsViaExternalizeModuleParameters(self):
        mdl = SimpleParams()
        externalize_module_parameters(mdl)

        fxb = FxProgramsBuilder(mdl)

        @fxb.export_program(
            args=(torch.empty([128, 20]),),
        )
        def _compute1(module, x):
            return module.forward(x)

        @fxb.export_program(
            args=(torch.empty([128, 20]),),
        )
        def _compute2(module, x):
            return module.forward(x)

        exported = export(fxb)
        module_str = str(exported.mlir_module)
        print(module_str)
        self.assertIn("util.global private @__auto.classifier.weight", module_str)
        self.assertIn("util.global private @__auto.classifier.bias", module_str)

        # It's clunky to verify ordering, but we explicitly guarantee that
        # implicitly exported globals are emitted in order of declaration,
        # preceding all functions.
        g1_index = module_str.index("util.global private @__auto.classifier.weight")
        g2_index = module_str.index("util.global private @__auto.classifier.bias")
        f_index = module_str.index("func")
        self.assertGreater(g2_index, g1_index)
        self.assertGreater(f_index, g2_index)

        # Should only be two.
        self.assertEqual(2, module_str.count("util.global private"))
        # And two loads each loads.
        self.assertEqual(
            2, module_str.count("util.global.load @__auto.classifier.weight")
        )
        self.assertEqual(
            2, module_str.count("util.global.load @__auto.classifier.bias")
        )


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
