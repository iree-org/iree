# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test concat op."""

from absl import app
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf


class ConcatOpsModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([1, 5, 0], tf.float32),
      tf.TensorSpec([1, 5, 1], tf.float32),
  ])
  def concat_zero_dim(self, a, b):
    return tf.concat([a, b], axis=2)

  @tf.function(input_signature=[
      tf.TensorSpec([1, 5, 1], tf.float32),
      tf.TensorSpec([1, 5, 1], tf.float32),
  ])
  def concat0axis(self, a, b):
    return tf.concat([a, b], axis=0)

  @tf.function(input_signature=[
      tf.TensorSpec([1, 5, 1], tf.float32),
      tf.TensorSpec([1, 5, 1], tf.float32),
  ])
  def concat1axis(self, a, b):
    return tf.concat([a, b], axis=1)

  @tf.function(input_signature=[
      tf.TensorSpec([1, 5, 1], tf.float32),
      tf.TensorSpec([1, 5, 1], tf.float32),
  ])
  def concat2axis(self, a, b):
    return tf.concat([a, b], axis=2)


class ConcatOpsTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(ConcatOpsModule)

  def test_concat_zero_dim(self):

    def concat_zero_dim(module):
      a = tf_utils.uniform([1, 5, 0])
      b = tf_utils.uniform([1, 5, 1])
      module.concat_zero_dim(a, b)

    self.compare_backends(concat_zero_dim, self._modules)

  def test_concat0axis(self):

    def concat0axis(module):
      a = tf_utils.uniform([1, 5, 1])
      b = tf_utils.uniform([1, 5, 1])
      module.concat0axis(a, b)

    self.compare_backends(concat0axis, self._modules)

  def test_concat1axis(self):

    def concat1axis(module):
      a = tf_utils.uniform([1, 5, 1])
      b = tf_utils.uniform([1, 5, 1])
      module.concat1axis(a, b)

    self.compare_backends(concat1axis, self._modules)

  def test_concat2axis(self):

    def concat2axis(module):
      a = tf_utils.uniform([1, 5, 1])
      b = tf_utils.uniform([1, 5, 1])
      module.concat2axis(a, b)

    self.compare_backends(concat2axis, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
