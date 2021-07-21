# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from absl import app
from iree.tf.support import tf_test_utils
import numpy as np
import tensorflow.compat.v2 as tf


class ResourcesOpsModule(tf.Module):

  def __init__(self):
    super().__init__()
    self.counter = tf.Variable(0.0)

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def add_assign(self, value):
    return self.counter.assign_add(value)

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def set_value(self, new_value):
    self.counter.assign(new_value)

  @tf.function(input_signature=[])
  def get_value(self):
    return self.counter


class ResourcesOpsTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(ResourcesOpsModule)

  def test_add_assign(self):

    def add_assign(module):
      module.add_assign(np.array(9., dtype=np.float32))

    self.compare_backends(add_assign, self._modules)

  def test_assign_get(self):

    def assign_get(module):
      module.set_value(np.array(9., dtype=np.float32))
      return module.get_value()

    self.compare_backends(assign_get, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
