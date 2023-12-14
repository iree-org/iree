# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import tempfile
import unittest


def run_tool(*argv: str):
    try:
        from iree.compiler.tools.import_onnx import __main__

        args = __main__.parse_arguments(list(argv))
        __main__.main(args)
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"Tool exited with code {e.code}")


ONNX_FILE_PATH = os.path.join(os.path.dirname(__file__), "testdata", "LeakyReLU.onnx")


class IrToolTest(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.outputPath = f.name

    def tearDown(self) -> None:
        if os.path.exists(self.outputPath):
            os.unlink(self.outputPath)

    def testConsoleOutput(self):
        # Just test that it doesn't crash: rely on the file test for verification.
        run_tool(ONNX_FILE_PATH)

    def testDisableVerify(self):
        # Just test that the flag is accepted.
        run_tool(ONNX_FILE_PATH, "--no-verify")

    def testFileOutput(self):
        run_tool(ONNX_FILE_PATH, "-o", self.outputPath)
        with open(self.outputPath, "rt") as f:
            contents = f.read()
            self.assertIn("torch.operator", contents)


if __name__ == "__main__":
    try:
        import onnx
    except ModuleNotFoundError:
        print(f"Skipping test {__file__} because Python dependency `onnx` is not found")
        sys.exit(0)

    unittest.main()
