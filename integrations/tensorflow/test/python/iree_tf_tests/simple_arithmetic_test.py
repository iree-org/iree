# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Several baseline e2e simple arithmetic tests."""

from absl import app
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf


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


class SimpleArithmeticTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(SimpleArithmeticModule)

  def test_simple_mul(self):

    def simple_mul(module):
      a = np.array([1., 2., 3., 4.], dtype=np.float32)
      b = np.array([400., 5., 6., 7.], dtype=np.float32)
      c = module.simple_mul(a, b)
      module.simple_mul(a, c)

    self.compare_backends(simple_mul, self._modules)

  def test_simple_matmul(self):

    def simple_matmul(module):
      # Note: scaling by a small value to increase numerical stability.
      a = tf_utils.uniform((128, 3072)) * 1e-3
      b = tf_utils.uniform((3072, 256)) * 1e-3
      module.simple_matmul(a, b)

    self.compare_backends(simple_matmul, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
