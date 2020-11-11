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
import tensorflow.compat.v2 as tf


class BooleanModule(tf_test_utils.TestModule):

  @tf_test_utils.tf_function_unittest(input_signature=[])
  def constant(self):
    return np.array([True, False, True], dtype=np.bool)

  @tf_test_utils.tf_function_unittest(
      input_signature=[tf.TensorSpec([4], tf.float32)],
      input_args=[np.array([0.0, 1.2, 1.5, 3.75], dtype=np.float32)])
  def greater_than(self, x):
    return x > 1.0

  @tf_test_utils.tf_function_unittest(
      input_signature=[
          tf.TensorSpec([4], tf.bool),
          tf.TensorSpec([4], tf.bool)
      ],
      input_args=[
          np.array([True, True, False, False], dtype=np.bool),
          np.array([True, False, False, True], dtype=np.bool)
      ],
  )
  def logical_and(self, x, y):
    return tf.math.logical_and(x, y)


class BooleanTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(BooleanModule)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  BooleanTest.generate_unittests(BooleanModule)
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
