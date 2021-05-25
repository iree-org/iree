# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from absl import app
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf


class GatherModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4, 8], tf.float32),
      tf.TensorSpec([], tf.int32)
  ])
  def gather_axis0_scalar(self, params, indices):
    return tf.gather(params, indices)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 8], tf.float32),
      tf.TensorSpec([2], tf.int32)
  ])
  def gather_axis0_batch0(self, params, indices):
    return tf.gather(params, indices)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 7, 8], tf.float32),
      tf.TensorSpec([2], tf.int32)
  ])
  def gather_axis1_batch0(self, params, indices):
    return tf.gather(params, indices, axis=1)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 7, 8, 2], tf.float32),
      tf.TensorSpec([4, 1], tf.int32)
  ])
  def gather_axis2_batch1(self, params, indices):
    return tf.gather(params, indices, axis=2, batch_dims=1)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 7, 8, 2], tf.float32),
      tf.TensorSpec([4, 1], tf.int32)
  ])
  def gather_axis1_batch1(self, params, indices):
    return tf.gather(params, indices, axis=1, batch_dims=1)

  @tf.function(input_signature=[
      tf.TensorSpec([2, 4], tf.int32),
      tf.TensorSpec([2, 4], tf.int32)
  ])
  def gather_axis2_batch2(self, params, indices):
    return tf.gather(params, indices, axis=1, batch_dims=1)


class GatherTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(GatherModule)

  # yapf: disable
  def test_gather_axis0_scalar(self):
    def gather_axis0_scalar(module):
      indices = np.array(2, dtype=np.int32)
      params = tf_utils.ndarange([4, 8])
      module.gather_axis0_scalar(params, indices)
    self.compare_backends(gather_axis0_scalar, self._modules)

  def test_gather_axis0_batch0(self):
    def gather_axis0_batch0(module):
      indices = np.array([2, 3], dtype=np.int32)
      params = tf_utils.ndarange([4, 8])
      module.gather_axis0_batch0(params, indices)
    self.compare_backends(gather_axis0_batch0, self._modules)

  def test_gather_axis1_batch0(self):
    def gather_axis1_batch0(module):
      indices = np.array([2, 3], dtype=np.int32)
      params = tf_utils.ndarange([4, 7, 8])
      module.gather_axis1_batch0(params, indices)
    self.compare_backends(gather_axis1_batch0, self._modules)

  def test_gather_axis2_batch1(self):
    def gather_axis2_batch1(module):
      indices = np.array([[2], [3], [0], [1]], dtype=np.int32)
      params = tf_utils.ndarange([4, 7, 8, 2])
      module.gather_axis2_batch1(params, indices)
    self.compare_backends(gather_axis2_batch1, self._modules)

  def test_gather_axis1_batch1(self):
    def gather_axis1_batch1(module):
      indices = np.array([[2], [3], [0], [1]], dtype=np.int32)
      params = tf_utils.ndarange([4, 7, 8, 2])
      module.gather_axis1_batch1(params, indices)
    self.compare_backends(gather_axis1_batch1, self._modules)

  def test_gather_axis2_batch2(self):
    def gather_axis2_batch2(module):
      indices = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32)
      values = np.array([[0, 1, 2, 3], [9, 8, 7, 0]], dtype=np.int32)
      module.gather_axis2_batch2(values, indices)
    self.compare_backends(gather_axis2_batch2, self._modules)
  # yapf: enable


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
