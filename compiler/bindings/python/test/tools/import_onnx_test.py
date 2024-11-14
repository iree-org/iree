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
LARGE_WEIGHTS_ONNX_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "testdata", "conv.onnx"
)


class ImportOnnxTest(unittest.TestCase):
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


class ImportOnnxwithExternalizationTest(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.outputPath = f.name

    def tearDown(self) -> None:
        if os.path.exists(self.outputPath):
            os.unlink(self.outputPath)
        if os.path.exists("custom_params_file.irpa"):
            os.unlink("custom_params_file.irpa")
        if os.path.exists(str(self.outputPath) + "_params.irpa"):
            os.unlink(str(self.outputPath) + "_params.irpa")

    def testExternalizeWeightsDefaultThreshold(self):
        run_tool(
            LARGE_WEIGHTS_ONNX_FILE_PATH, "--externalize-params", "-o", self.outputPath
        )
        with open(self.outputPath, "rt") as f:
            contents = f.read()
            self.assertIn("util.global", contents)
            self.assertIn("util.global.load", contents)
            # The bias is smaller in volume than the default 100 elements,
            # so it should still be inlined.
            self.assertIn("onnx.Constant", contents)
        assert os.path.isfile(str(self.outputPath) + "_params.irpa")

    def testExternalizeParamsSaveCustomPath(self):
        run_tool(
            LARGE_WEIGHTS_ONNX_FILE_PATH,
            "--externalize-params",
            "--save-params-to",
            "custom_params_file.irpa",
            "-o",
            self.outputPath,
        )
        with open(self.outputPath, "rt") as f:
            contents = f.read()
            self.assertIn("util.global", contents)
            self.assertIn("util.global.load", contents)
        assert os.path.isfile("custom_params_file.irpa")

    def testExternalizeTooHighThreshold(self):
        num_elements_weights = 1 * 256 * 100 * 100 + 1
        run_tool(
            LARGE_WEIGHTS_ONNX_FILE_PATH,
            "--externalize-params",
            "--num-elements-threshold",
            str(num_elements_weights),
            "-o",
            self.outputPath,
        )
        with open(self.outputPath, "rt") as f:
            contents = f.read()
            self.assertNotIn("util.global", contents)
            self.assertNotIn("util.global.load", contents)
            self.assertIn("onnx.Constant", contents)

    def testExternalizeMinimumThreshold(self):
        run_tool(
            LARGE_WEIGHTS_ONNX_FILE_PATH,
            "--externalize-params",
            "--num-elements-threshold",
            "0",
            "-o",
            self.outputPath,
        )
        with open(self.outputPath, "rt") as f:
            contents = f.read()
            self.assertIn("util.global", contents)
            self.assertIn("util.global.load", contents)
            # As the max allowed number of elements for inlining is 0 elements,
            # there should be no inlined constants.
            self.assertNotIn("onnx.Constant", contents)


if __name__ == "__main__":
    try:
        import onnx
    except ModuleNotFoundError:
        print(f"Skipping test {__file__} because Python dependency `onnx` is not found")
        sys.exit(0)

    unittest.main()
