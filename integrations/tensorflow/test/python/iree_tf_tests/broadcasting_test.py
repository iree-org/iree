# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test broadcasting support."""

from absl import app
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf


class BroadcastingModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([None], tf.float32),
      tf.TensorSpec([None], tf.float32),
  ])
  def add(self, lhs, rhs):
    return lhs + rhs


class BroadcastingTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(BroadcastingModule)

  def test_add_same_shape(self):

    def add_same_shape(module):
      lhs = tf_utils.uniform([4])
      rhs = tf_utils.uniform([4])
      module.add(lhs, rhs)

    self.compare_backends(add_same_shape, self._modules)

  def test_add_broadcast_lhs(self):

    def add_broadcast_lhs(module):
      lhs = tf_utils.uniform([1])
      rhs = tf_utils.uniform([4])
      module.add(lhs, rhs)

    self.compare_backends(add_broadcast_lhs, self._modules)

  def test_add_broadcast_rhs(self):

    def add_broadcast_rhs(module):
      lhs = tf_utils.uniform([4])
      rhs = tf_utils.uniform([1])
      module.add(lhs, rhs)

    self.compare_backends(add_broadcast_rhs, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
