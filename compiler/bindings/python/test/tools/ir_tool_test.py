# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler.tools.ir_tool import __main__

import os
import tempfile
import unittest


def run_tool(*argv: str):
    try:
        args = __main__.parse_arguments(list(argv))
        __main__.main(args)
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"Tool exited with code {e.code}")


class IrToolTest(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.inputPath = f.name
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.outputPath = f.name

    def tearDown(self) -> None:
        if os.path.exists(self.inputPath):
            os.unlink(self.inputPath)
        if os.path.exists(self.outputPath):
            os.unlink(self.outputPath)

    def saveInput(self, contents, text=True):
        with open(self.inputPath, "wt" if text else "wb") as f:
            f.write(contents)

    def loadOutput(self, text=True):
        with open(self.outputPath, "rt" if text else "rb") as f:
            return f.read()

    def testCpDefaultArgs(self):
        self.saveInput(
            r"""
            builtin.module {
            }
            """
        )
        run_tool("copy", self.inputPath, "-o", self.outputPath)
        output = self.loadOutput()
        print("Output:", output)
        self.assertIn("module", output)

    def testCpEmitBytecode(self):
        self.saveInput(
            r"""
            builtin.module {
            }
            """
        )
        run_tool(
            "copy",
            "--emit-bytecode",
            self.inputPath,
            "-o",
            self.outputPath,
        )
        output = self.loadOutput(text=False)
        self.assertIn(b"MLIR", output)

    def testCpEmitBytecodeVersion(self):
        self.saveInput(
            r"""
            builtin.module {
            }
            """
        )
        run_tool(
            "copy",
            "--emit-bytecode",
            "--bytecode-version=0",
            self.inputPath,
            "-o",
            self.outputPath,
        )
        output = self.loadOutput(text=False)
        self.assertIn(b"MLIR", output)

    def testStripDataWithImport(self):
        self.saveInput(
            r"""
            builtin.module {
                func.func @main() -> tensor<4xf32> {
                    %0 = arith.constant dense<[0.1, 0.2, 0.3, 0.4]> : tensor<4xf32>
                    func.return %0 : tensor<4xf32>
                }
            }
            """
        )
        run_tool("strip-data", self.inputPath, "-o", self.outputPath)
        output = self.loadOutput()
        print("Output:", output)
        self.assertIn("#util.byte_pattern", output)

    def testStripDataNoImport(self):
        # Without import, ml_program.global is not recognized.
        self.saveInput(
            r"""
            builtin.module {
                ml_program.global public @foobar(dense<[0.1, 0.2, 0.3, 0.4]> : tensor<4xf32>) : tensor<4xf32>
            }
            """
        )
        run_tool("strip-data", "--no-import", self.inputPath, "-o", self.outputPath)
        output = self.loadOutput()
        print("Output:", output)
        self.assertNotIn("#util.byte_pattern", output)

    def testStripDataParseError(self):
        self.saveInput(
            r"""
            FOOBAR
            """
        )
        with self.assertRaisesRegex(RuntimeError, "Error parsing source file"):
            run_tool("strip-data", self.inputPath, "-o", self.outputPath)

    def testStripDataEmitBytecode(self):
        self.saveInput(
            r"""
            builtin.module {
            }
            """
        )
        run_tool("strip-data", "--emit-bytecode", self.inputPath, "-o", self.outputPath)
        output = self.loadOutput(text=False)
        self.assertIn(b"MLIR", output)

    def testStripDataEmitBytecodeVersion(self):
        self.saveInput(
            r"""
            builtin.module {
            }
            """
        )
        run_tool(
            "strip-data",
            "--emit-bytecode",
            "--bytecode-version=0",
            self.inputPath,
            "-o",
            self.outputPath,
        )
        output = self.loadOutput(text=False)
        self.assertIn(b"MLIR", output)


if __name__ == "__main__":
    unittest.main()
