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

import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class MathModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([4], tf.float32)])
  def abs(self, x):
    return tf.math.abs(x)

  @tf.function(input_signature=[tf.TensorSpec([4], tf.float32)])
  def cos(self, x):
    return tf.math.cos(x)

  @tf.function(input_signature=[tf.TensorSpec([4], tf.float32)])
  def log(self, x):
    return tf.math.log(x)

  @tf.function(input_signature=[tf.TensorSpec([4], tf.float32)])
  def mod(self, x):
    return tf.math.mod(x, 2.0)


@tf_test_utils.compile_module(MathModule)
class MathTest(tf_test_utils.SavedModelTestCase):

  def test_abs(self):
    a = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    r = self.get_module().abs(a)
    r.print().assert_all_close()

  def test_cos(self):
    a = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    r = self.get_module().cos(a)
    r.print().assert_all_close()

  def test_log(self):
    a = np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float32)
    r = self.get_module().log(a)
    r.print().assert_all_close()

  def test_mod(self):
    a = np.array([0.0, 1.2, 1.5, 3.75], dtype=np.float32)
    r = self.get_module().mod(a)
    r.print().assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
