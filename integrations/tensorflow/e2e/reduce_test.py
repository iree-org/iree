# Lint as: python3
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
"""Tests for ops in the tf.math module."""

from absl import app
import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf


class ReduceModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([4, 4, 4], tf.float32)])
  def max(self, x):
    return tf.math.reduce_max(x, axis=1)

  @tf.function(input_signature=[tf.TensorSpec([4, 4, 4], tf.float32)])
  def min(self, x):
    return tf.math.reduce_min(x, axis=1)

  @tf.function(input_signature=[tf.TensorSpec([4, 4, 4], tf.float32)])
  def sum(self, x):
    return tf.math.reduce_sum(x, axis=1)

  @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.bool)])
  def reduce_any(self, x):
    return tf.math.reduce_any(x, axis=1)

  @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.bool)])
  def reduce_all(self, x):
    return tf.math.reduce_all(x, axis=1)


class ReduceTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(ReduceModule)

  # yapf: disable
  def test_max(self):
    def max(module):
      arr = tf_utils.uniform([4, 4, 4], dtype=tf.float32)
      module.max(arr)
    self.compare_backends(max, self._modules)

  def test_min(self):
    def min(module):
      arr = tf_utils.uniform([4, 4, 4], dtype=tf.float32)
      module.min(arr)
    self.compare_backends(min, self._modules)

  def test_sum(self):
    def sum(module):
      arr = tf_utils.uniform([4, 4, 4], dtype=tf.float32)
      module.sum(arr)
    self.compare_backends(sum, self._modules)

  def test_any(self):
    def reduce_any(module):
      arr = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.bool)
      module.reduce_any(arr)
    self.compare_backends(reduce_any, self._modules)

  def test_all(self):
    def reduce_all(module):
      arr = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.bool)
      module.reduce_all(arr)
    self.compare_backends(reduce_all, self._modules)
  # yapf: enable


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
