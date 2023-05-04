# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from absl import app
from iree.tf.support import tf_test_utils
import numpy as np
import tensorflow.compat.v2 as tf


class SimpleStatefulModule(tf.Module):

  def __init__(self):
    super().__init__()
    self.counter = tf.Variable(0.0)

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def inc_by(self, x):
    self.counter.assign(self.counter + x)

  @tf.function(input_signature=[])
  def get_state(self):
    return self.counter


class StatefulTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(SimpleStatefulModule)

  def test_stateful(self):

    def get_state(module):
      module.inc_by(np.array(1., dtype=np.float32))
      module.get_state()

    self.compare_backends(get_state, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
