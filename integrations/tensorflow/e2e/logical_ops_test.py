# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ops in the tf.math module that specifically handle logical ops."""

from absl import app
import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class LogicalOpsModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.bool),
      tf.TensorSpec([4], tf.bool)
  ])
  def logical_and(self, x, y):
    return tf.math.logical_and(x, y)

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.bool),
      tf.TensorSpec([4], tf.bool)
  ])
  def logical_or(self, x, y):
    return tf.math.logical_or(x, y)

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.bool),
      tf.TensorSpec([4], tf.bool)
  ])
  def logical_xor(self, x, y):
    return tf.math.logical_xor(x, y)

  @tf.function(input_signature=[tf.TensorSpec([4], tf.bool)])
  def logical_not(self, x):
    return tf.math.logical_not(x)


class LogicalOpsTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(LogicalOpsModule)

  # yapf: disable
  def test_logical_and(self):
    def logical_and(module):
      module.logical_and(
          np.array([1, 1, 0, 0], dtype=np.bool),
          np.array([0, 1, 1, 0], dtype=np.bool))
    self.compare_backends(logical_and, self._modules)

  def test_logical_or(self):
    def logical_or(module):
      module.logical_or(
          np.array([1, 1, 0, 0], dtype=np.bool),
          np.array([0, 1, 1, 0], dtype=np.bool))
    self.compare_backends(logical_or, self._modules)

  def test_logical_xor(self):
    def logical_xor(module):
      module.logical_xor(
          np.array([1, 1, 0, 0], dtype=np.bool),
          np.array([0, 1, 1, 0], dtype=np.bool))
    self.compare_backends(logical_xor, self._modules)

  def test_logical_not(self):
    def logical_not(module):
      module.logical_not(np.array([0, 1, 1, 0], dtype=np.bool))
    self.compare_backends(logical_not, self._modules)
  # yapf: enable


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
