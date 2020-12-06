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

from pyiree.compiler2.tf import *

if not is_available():
  print(f"Skipping test {__file__} because TensorFlow is not installed")
  sys.exit(0)

import tensorflow as tf


class SimpleArithmeticModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    return a * b

  @tf.function(input_signature=[
      tf.TensorSpec([128, 3072], tf.float32),
      tf.TensorSpec([3072, 256], tf.float32),
  ])
  def simple_matmul(self, a, b):
    return tf.matmul(a, b)


class TfCompilerTest(unittest.TestCase):

  def testImportSavedModel(self):
    import_mlir = compile_saved_model(self.smdir,
                                      import_only=True).decode("utf-8")
    self.assertIn("func @simple_matmul", import_mlir)

  def testCompileSavedModel(self):
    binary = compile_saved_model(self.smdir,
                                 target_backends=DEFAULT_TESTING_BACKENDS)
    logging.info("Compiled len: %d", len(binary))
    self.assertIn(b"simple_matmul", binary)
    self.assertIn(b"simple_mul", binary)

  def testCompileModule(self):
    binary = compile_module(self.m, target_backends=DEFAULT_TESTING_BACKENDS)
    logging.info("Compiled len: %d", len(binary))
    self.assertIn(b"simple_matmul", binary)
    self.assertIn(b"simple_mul", binary)

  @classmethod
  def setUpClass(cls):
    cls.m = SimpleArithmeticModule()
    cls.tempdir = tempfile.TemporaryDirectory()
    cls.smdir = os.path.join(cls.tempdir.name, "arith.sm")
    tf.saved_model.save(
        cls.m,
        cls.smdir,
        options=tf.saved_model.SaveOptions(save_debug_info=True))

  @classmethod
  def tearDownClass(cls):
    cls.tempdir.cleanup()


if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  unittest.main()
