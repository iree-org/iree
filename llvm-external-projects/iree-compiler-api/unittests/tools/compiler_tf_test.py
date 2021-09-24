# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
import sys
import tempfile
import unittest

# TODO: No idea why pytype cannot find names from this module.
# pytype: disable=name-error
import iree.compiler.tools.tf

if not iree.compiler.tools.tf.is_available():
  print(f"Skipping test {__file__} because the IREE TensorFlow compiler "
        f"is not installed")
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


# TODO(laurenzo): More test cases needed (may need additional files).
# Specifically, figure out how to test v1 models.
class TfCompilerTest(tf.test.TestCase):

  def testImportSavedModel(self):
    import_mlir = iree.compiler.tools.tf.compile_saved_model(
        self.smdir, import_only=True, output_generic_mlir=True).decode("utf-8")
    self.assertIn("sym_name = \"simple_matmul\"", import_mlir)

  def testCompileSavedModel(self):
    binary = iree.compiler.tools.tf.compile_saved_model(
        self.smdir,
        target_backends=iree.compiler.tools.tf.DEFAULT_TESTING_BACKENDS)
    logging.info("Compiled len: %d", len(binary))
    self.assertIn(b"simple_matmul", binary)
    self.assertIn(b"simple_mul", binary)

  def testCompileModule(self):
    binary = iree.compiler.tools.tf.compile_module(
        self.m, target_backends=iree.compiler.tools.tf.DEFAULT_TESTING_BACKENDS)
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
  tf.test.main()
