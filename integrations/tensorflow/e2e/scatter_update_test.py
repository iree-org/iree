# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test scatter update behavior for tensorflow."""

from absl import app
from iree.tf.support import tf_test_utils
import numpy as np
import tensorflow.compat.v2 as tf


class ScatterUpdateModule(tf.Module):

  def __init__(self):
    pass

  @tf.function(input_signature=[
      tf.TensorSpec([8], tf.int32),
      tf.TensorSpec([3, 1], tf.int32),
      tf.TensorSpec([3], tf.int32)
  ])
  def scatter_update_1D(self, tensor, indices, updates):
    return tf.tensor_scatter_nd_update(tensor, indices, updates)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 3], tf.int32),
      tf.TensorSpec([3, 2], tf.int32),
      tf.TensorSpec([3], tf.int32)
  ])
  def scatter_update_2D(self, tensor, indices, updates):
    return tf.tensor_scatter_nd_update(tensor, indices, updates)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 3], tf.int32),
      tf.TensorSpec([1, 1], tf.int32),
      tf.TensorSpec([1, 3], tf.int32)
  ])
  def scatter_update_2D_slice(self, tensor, indices, updates):
    return tf.tensor_scatter_nd_update(tensor, indices, updates)


class ScatterUpdateTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(ScatterUpdateModule)

  # yapf: disable
  def test_scatter_update_1D(self):
    def scatter_update_1D(module):
      tensor = np.ones([8], dtype=np.int32)
      indices = np.array([[4], [5], [6]], dtype=np.int32)
      updates = np.array([9, 10, 11], dtype=np.int32)
      module.scatter_update_1D(tensor, indices, updates)
    self.compare_backends(scatter_update_1D, self._modules)

  def test_scatter_update_2D(self):
    def scatter_update_2D(module):
      tensor = np.ones([4, 3], dtype=np.int32)
      indices = np.array([[1, 0], [2, 1], [3, 2]], dtype=np.int32)
      updates = np.array([2, 5, 8], dtype=np.int32)
      module.scatter_update_2D(tensor, indices, updates)
    self.compare_backends(scatter_update_2D, self._modules)

  def test_scatter_update_2D_slice(self):
    def scatter_update_2D_slice(module):
      tensor = np.ones([4, 3], dtype=np.int32)
      indices = np.array([[1]], dtype=np.int32)
      updates = np.array([[2, 3, 4]], dtype=np.int32)
      module.scatter_update_2D_slice(tensor, indices, updates)
    self.compare_backends(scatter_update_2D_slice, self._modules)
  # yapf: enable


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
