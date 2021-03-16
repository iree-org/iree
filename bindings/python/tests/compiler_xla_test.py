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
from pyiree.compiler.xla import *

if not is_available():
  print(f"Skipping test {__file__} because the IREE XLA compiler "
        f"is not installed")
  sys.exit(0)


class CompilerTest(unittest.TestCase):

  def testImportBinaryPbFile(self):
    path = os.path.join(os.path.dirname(__file__), "testdata", "xla_sample.pb")
    text = compile_file(path, import_only=True).decode("utf-8")
    logging.info("%s", text)
    self.assertIn("mhlo.constant", text)
    self.assertIn("iree.module.export", text)

  def testCompileBinaryPbFile(self):
    path = os.path.join(os.path.dirname(__file__), "testdata", "xla_sample.pb")
    binary = compile_file(path, target_backends=DEFAULT_TESTING_BACKENDS)
    logging.info("Binary length = %d", len(binary))
    self.assertIn(b"main", binary)

  def testImportBinaryPbFileOutputFile(self):
    path = os.path.join(os.path.dirname(__file__), "testdata", "xla_sample.pb")
    with tempfile.NamedTemporaryFile("wt", delete=False) as f:
      try:
        f.close()
        output = compile_file(path, import_only=True, output_file=f.name)
        self.assertIsNone(output)
        with open(f.name, "rt") as f_read:
          text = f_read.read()
      finally:
        os.remove(f.name)
    logging.info("%s", text)
    self.assertIn("mhlo.constant", text)

  def testCompileBinaryPbFileOutputFile(self):
    path = os.path.join(os.path.dirname(__file__), "testdata", "xla_sample.pb")
    with tempfile.NamedTemporaryFile("wt", delete=False) as f:
      try:
        f.close()
        output = compile_file(path,
                              output_file=f.name,
                              target_backends=DEFAULT_TESTING_BACKENDS)
        self.assertIsNone(output)
        with open(f.name, "rb") as f_read:
          binary = f_read.read()
      finally:
        os.remove(f.name)
    logging.info("Binary length = %d", len(binary))
    self.assertIn(b"main", binary)

  def testImportBinaryPbBytes(self):
    path = os.path.join(os.path.dirname(__file__), "testdata", "xla_sample.pb")
    with open(path, "rb") as f:
      content = f.read()
    text = compile_str(content, import_only=True).decode("utf-8")
    logging.info("%s", text)
    self.assertIn("mhlo.constant", text)

  def testCompileBinaryPbBytes(self):
    path = os.path.join(os.path.dirname(__file__), "testdata", "xla_sample.pb")
    with open(path, "rb") as f:
      content = f.read()
    binary = compile_str(content, target_backends=DEFAULT_TESTING_BACKENDS)
    logging.info("Binary length = %d", len(binary))
    self.assertIn(b"main", binary)

  def testImportHloTextFile(self):
    path = os.path.join(os.path.dirname(__file__), "testdata", "xla_sample.hlo")
    text = compile_file(path, import_only=True,
                        import_format="hlo_text").decode("utf-8")
    logging.info("%s", text)
    self.assertIn("mhlo.constant", text)
    self.assertIn("iree.module.export", text)

  def testImportHloTextStr(self):
    path = os.path.join(os.path.dirname(__file__), "testdata", "xla_sample.hlo")
    with open(path, "rt") as f:
      content = f.read()
    text = compile_str(content, import_only=True,
                       import_format="hlo_text").decode("utf-8")
    logging.info("%s", text)
    self.assertIn("mhlo.constant", text)
    self.assertIn("iree.module.export", text)

  def testImportHloTextBytes(self):
    path = os.path.join(os.path.dirname(__file__), "testdata", "xla_sample.hlo")
    with open(path, "rb") as f:
      content = f.read()
    text = compile_str(content, import_only=True,
                       import_format="hlo_text").decode("utf-8")
    logging.info("%s", text)
    self.assertIn("mhlo.constant", text)
    self.assertIn("iree.module.export", text)


if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  unittest.main()
