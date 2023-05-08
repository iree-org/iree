# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Batch norm tests."""

from absl import app
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf


class BatchNormModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4, 16], tf.float32),
      tf.TensorSpec([16], tf.float32),
      tf.TensorSpec([16], tf.float32),
      tf.TensorSpec([16], tf.float32),
      tf.TensorSpec([16], tf.float32),
  ])
  def batch_norm_inference(self, x, mean, variance, offset, scale):
    return tf.nn.batch_normalization(x,
                                     mean=mean,
                                     variance=variance,
                                     offset=offset,
                                     scale=scale,
                                     variance_epsilon=1e-4)


class BatchNormTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(BatchNormModule)

  def test_batch_norm_inference(self):

    def batch_norm_inference(module):
      # Note: scaling by a small value to increase numerical stability.
      x = tf_utils.uniform((4, 16)) * 1e-3
      mean = tf_utils.uniform((16,)) * 1e-3
      variance = tf_utils.uniform((16,), low=0.0) * 1e-3
      offset = tf_utils.uniform((16,)) * 1e-3
      scale = tf_utils.uniform((16,)) * 1e-3
      module.batch_norm_inference(x, mean, variance, offset, scale)

    self.compare_backends(batch_norm_inference, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
