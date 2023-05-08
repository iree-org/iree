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

HIDDEN_1_DIM = 256
HIDDEN_2_DIM = 256
INPUT_DIM = 728  # 28 * 28
CLASSES = 10


class DynamicMlpModule(tf.Module):

  def __init__(self,
               hidden_1_dim=256,
               hidden_2_dim=256,
               input_dim=28 * 28,
               classes=10):
    super().__init__()
    tf_utils.set_random_seed()
    self.hidden_1_dim = hidden_1_dim
    self.hidden_2_dim = hidden_2_dim
    self.input_dim = input_dim
    self.classes = classes
    self.h1_weights = tf.Variable(tf.random.normal([input_dim, hidden_1_dim]))
    self.h2_weights = tf.Variable(tf.random.normal([hidden_1_dim,
                                                    hidden_2_dim]))
    self.out_weights = tf.Variable(tf.random.normal([hidden_2_dim, classes]))
    self.h1_bias = tf.Variable(tf.random.normal([hidden_1_dim]))
    self.h2_bias = tf.Variable(tf.random.normal([hidden_2_dim]))
    self.out_bias = tf.Variable(tf.random.normal([classes]))

    # Compile with dynamic batch dim.
    self.predict = tf.function(
        input_signature=[tf.TensorSpec([None, self.input_dim])])(self.predict)

  def mlp(self, x):
    layer_1 = tf.sigmoid(tf.add(tf.matmul(x, self.h1_weights), self.h1_bias))
    layer_2 = tf.sigmoid(
        tf.add(tf.matmul(layer_1, self.h2_weights), self.h2_bias))
    return tf.sigmoid(
        tf.add(tf.matmul(layer_2, self.out_weights), self.out_bias))

  def predict(self, x):
    return tf.nn.softmax(self.mlp(x))


class DynamicMlpTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(DynamicMlpModule,
                                                    exported_names=["predict"])

  def test_dynamic_batch(self):

    def dynamic_batch(module):
      x = tf_utils.uniform([3, 28 * 28]) * 1e-3
      module.predict(x)

    self.compare_backends(dynamic_batch, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
