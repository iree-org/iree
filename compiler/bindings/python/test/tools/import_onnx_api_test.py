# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import tempfile
import unittest

from iree.compiler.tools.onnx import compile_model
from iree.compiler.tools import OutputFormat

ONNX_FILE_PATH = os.path.join(os.path.dirname(
    __file__), "testdata", "LeakyReLU.onnx")


class ImportOnnxTest(unittest.TestCase):
    def setUp(self):
        self.model = onnx.load(ONNX_FILE_PATH)

    def tearDown(self) -> None:
        return

    def testImport(self):
        module = compile_model(self.model, import_only=True,
                               verify_module=True, use_bytecode=False)
        self.assertIsNotNone(module)
        self.assertIn("module", module)

    def testCompile(self):
        module = compile_model(self.model, import_only=False, verify_module=True, use_bytecode=False, target_backends=[
                               "llvm-cpu"], output_format=OutputFormat.MLIR_TEXT)
        self.assertIsNotNone(module)
        self.assertIn(b"module", module)

    def testDisableVerify(self):
        module = compile_model(self.model, import_only=True,
                               verify_module=False, use_bytecode=False)
        self.assertIsNotNone(module)

    def testFileOutput(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            compile_model(self.model, output_file=tmpdir + "/" + "test.mlir",
                          import_only=True, verify_module=True, use_bytecode=False)
            with open(tmpdir + "/" + "test.mlir", "rt", encoding="utf-8") as f:
                contents = f.read()
            self.assertIn("torch.operator", contents)

    def testIntermediateFiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            compile_model(self.model, output_file=tmpdir + "/" + "test.mlir", save_temp_iree_input=tmpdir + "/" +
                          "temp.mlir", import_only=False, verify_module=True, use_bytecode=False, target_backends=["llvm-cpu"])
            self.assertTrue(os.path.exists(tmpdir + "/" + "test.mlir"))
            self.assertTrue(os.path.exists(tmpdir + "/" + "temp.mlir"))

    def testBytecode(self):
        module = compile_model(self.model, import_only=True,
                               verify_module=True, use_bytecode=True)
        self.assertIsNotNone(module)

    def testEntryPointName(self):
        module = compile_model(self.model, import_only=True, verify_module=True,
                               use_bytecode=False, entry_point_name="testEntryPoint")
        self.assertIsNotNone(module)
        self.assertIn("func.func @testEntryPoint", module)

    def testModuleName(self):
        module = compile_model(self.model, import_only=True, verify_module=True,
                               use_bytecode=False, module_name="TestModule")
        self.assertIsNotNone(module)
        self.assertIn("@TestModule", module)


if __name__ == "__main__":
    try:
        import onnx
    except ModuleNotFoundError:
        print(
            f"Skipping test {__file__} because Python dependency `onnx` is not found")
        sys.exit(0)

    unittest.main()
