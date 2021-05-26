# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from absl import app
from iree.tf.support import tf_test_utils
import numpy as np
import tensorflow.compat.v2 as tf


class BroadcastToModule(tf.Module):

  def __init__(self):
    pass

  @tf.function(input_signature=[
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([2], tf.int32)
  ])
  def scalar_broadcast_to(self, x, shape):
    return tf.broadcast_to(x, shape)


class BroadcastToTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(BroadcastToModule)

  def test_scalar_broadcast_to(self):

    def scalar_broadcast_to(module):
      x = np.array(1, dtype=np.float32)
      shape = np.array([3, 3], dtype=np.int32)
      result = module.scalar_broadcast_to(x, shape)

    self.compare_backends(scalar_broadcast_to, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
