# Lint as: python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging
import os
import io
import tempfile
import unittest

import iree.compiler

SIMPLE_MUL_ASM = """
func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
      attributes { iree.module.export } {
    %0 = "mhlo.multiply"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
"""


class CompilerTest(unittest.TestCase):

  def testNoTargetBackends(self):
    with self.assertRaisesRegex(
        ValueError, "Expected a non-empty list for 'target_backends'"):
      binary = iree.compiler.compile_str(SIMPLE_MUL_ASM)

  def testCompileStr(self):
    binary = iree.compiler.compile_str(
        SIMPLE_MUL_ASM, target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS)
    logging.info("Flatbuffer size = %d", len(binary))
    self.assertTrue(binary)

  # Compiling the string form means that the compiler does not have a valid
  # source file name, which can cause issues on the AOT side. Verify
  # specifically. See: https://github.com/google/iree/issues/4439
  def testCompileStrLLVMAOT(self):
    binary = iree.compiler.compile_str(SIMPLE_MUL_ASM,
                                       target_backends=["dylib-llvm-aot"])
    logging.info("Flatbuffer size = %d", len(binary))
    self.assertTrue(binary)

  # Verifies that multiple target_backends are accepted. Which two are not
  # load bearing.
  # See: https://github.com/google/iree/issues/4436
  def testCompileMultipleBackends(self):
    binary = iree.compiler.compile_str(
        SIMPLE_MUL_ASM, target_backends=["dylib-llvm-aot", "vulkan-spirv"])
    logging.info("Flatbuffer size = %d", len(binary))
    self.assertTrue(binary)

  def testCompileInputFile(self):
    with tempfile.NamedTemporaryFile("wt", delete=False) as f:
      try:
        f.write(SIMPLE_MUL_ASM)
        f.close()
        binary = iree.compiler.compile_file(
            f.name, target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS)
      finally:
        os.remove(f.name)
    logging.info("Flatbuffer size = %d", len(binary))
    self.assertIn(b"simple_mul", binary)

  def testCompileOutputFile(self):
    with tempfile.NamedTemporaryFile("wt", delete=False) as f:
      try:
        f.close()
        output = iree.compiler.compile_str(
            SIMPLE_MUL_ASM,
            output_file=f.name,
            target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS)
        self.assertIsNone(output)

        with open(f.name, "rb") as f_read:
          binary = f_read.read()
      finally:
        os.remove(f.name)
    logging.info("Flatbuffer size = %d", len(binary))
    self.assertIn(b"simple_mul", binary)

  def testOutputFbText(self):
    text = iree.compiler.compile_str(
        SIMPLE_MUL_ASM,
        output_format=iree.compiler.OutputFormat.FLATBUFFER_TEXT,
        target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS).decode("utf-8")
    # Just check for an arbitrary JSON-tag.
    self.assertIn('"exported_functions"', text)

  def testBadOutputFormat(self):
    with self.assertRaisesRegex(
        ValueError, "For output_format= argument, expected one of: "
        "FLATBUFFER_BINARY, FLATBUFFER_TEXT, MLIR_TEXT"):
      _ = iree.compiler.compile_str(
          SIMPLE_MUL_ASM,
          output_format="foobar",
          target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS)

  def testOutputFbTextParsed(self):
    text = iree.compiler.compile_str(
        SIMPLE_MUL_ASM,
        output_format='flatbuffer_text',
        target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS).decode("utf-8")
    # Just check for an arbitrary JSON-tag.
    self.assertIn('"exported_functions"', text)

  def testOutputMlirText(self):
    text = iree.compiler.compile_str(
        SIMPLE_MUL_ASM,
        output_format=iree.compiler.OutputFormat.MLIR_TEXT,
        target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS).decode("utf-8")
    # Just check for a textual op name.
    self.assertIn("vm.module", text)

  def testExtraArgsStderr(self):
    # pass-timing is not special: it just does something and emits to stderr.
    with io.StringIO() as buf, contextlib.redirect_stderr(buf):
      iree.compiler.compile_str(
          SIMPLE_MUL_ASM,
          extra_args=["--pass-timing"],
          target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS)
      stderr = buf.getvalue()
    self.assertIn("Pass execution timing report", stderr)

  def testAllOptions(self):
    binary = iree.compiler.compile_str(
        SIMPLE_MUL_ASM,
        optimize=False,
        strip_debug_ops=True,
        strip_source_map=True,
        strip_symbols=True,
        crash_reproducer_path="foobar.txt",
        enable_benchmark=True,
        target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS)

  def testException(self):
    with self.assertRaisesRegex(iree.compiler.CompilerToolError,
                                "Invoked with"):
      _ = iree.compiler.compile_str(
          "I'm a little teapot but not a valid program",
          target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS)


if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  unittest.main()
