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

import logging
import os
import sys
import tempfile
import unittest

# TODO: No idea why pytype cannot find names from this module.
# pytype: disable=name-error
import iree.compiler.tflite

if not iree.compiler.tflite.is_available():
  print(f"Skipping test {__file__} because the IREE TFLite compiler "
        f"is not installed")
  sys.exit(0)


class CompilerTest(unittest.TestCase):

  def testImportBinaryPbFile(self):
    path = os.path.join(os.path.dirname(__file__), "testdata",
                        "tflite_sample.fb")
    text = iree.compiler.tflite.compile_file(path,
                                             import_only=True).decode("utf-8")
    logging.info("%s", text)
    self.assertIn("tosa.mul", text)

  @unittest.skip("IREE tosa compilation not implemented yet")
  def testCompileBinaryPbFile(self):
    path = os.path.join(os.path.dirname(__file__), "testdata",
                        "tflite_sample.fb")
    binary = iree.compiler.tflite.compile_file(
        path, target_backends=iree.compiler.tflite.DEFAULT_TESTING_BACKENDS)
    logging.info("Binary length = %d", len(binary))
    self.assertIn(b"main", binary)

  def testImportBinaryPbFileOutputFile(self):
    path = os.path.join(os.path.dirname(__file__), "testdata",
                        "tflite_sample.fb")
    with tempfile.NamedTemporaryFile("wt", delete=False) as f:
      try:
        f.close()
        output = iree.compiler.tflite.compile_file(path,
                                                   import_only=True,
                                                   output_file=f.name)
        self.assertIsNone(output)
        with open(f.name, "rt") as f_read:
          text = f_read.read()
      finally:
        os.remove(f.name)
    logging.info("%s", text)
    self.assertIn("tosa.mul", text)

  @unittest.skip("IREE tosa compilation not implemented yet")
  def testCompileBinaryPbFileOutputFile(self):
    path = os.path.join(os.path.dirname(__file__), "testdata",
                        "tflite_sample.fb")
    with tempfile.NamedTemporaryFile("wt", delete=False) as f:
      try:
        f.close()
        output = iree.compiler.tflite.compile_file(
            path,
            output_file=f.name,
            target_backends=iree.compiler.tflite.DEFAULT_TESTING_BACKENDS)
        self.assertIsNone(output)
        with open(f.name, "rb") as f_read:
          binary = f_read.read()
      finally:
        os.remove(f.name)
    logging.info("Binary length = %d", len(binary))
    self.assertIn(b"main", binary)

  def testImportBinaryPbBytes(self):
    path = os.path.join(os.path.dirname(__file__), "testdata",
                        "tflite_sample.fb")
    with open(path, "rb") as f:
      content = f.read()
    text = iree.compiler.tflite.compile_str(content,
                                            import_only=True).decode("utf-8")
    logging.info("%s", text)
    self.assertIn("tosa.mul", text)

  @unittest.skip("IREE tosa compilation not implemented yet")
  def testCompileBinaryPbBytes(self):
    path = os.path.join(os.path.dirname(__file__), "testdata",
                        "tflite_sample.fb")
    with open(path, "rb") as f:
      content = f.read()
    binary = iree.compiler.tflite.compile_str(
        content, target_backends=iree.compiler.tflite.DEFAULT_TESTING_BACKENDS)
    logging.info("Binary length = %d", len(binary))
    self.assertIn(b"main", binary)


if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  unittest.main()
