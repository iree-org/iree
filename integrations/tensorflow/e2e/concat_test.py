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
"""Test concat op."""

from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf


class ConcatOpsModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([1, 5, 0], tf.float32),
      tf.TensorSpec([1, 5, 1], tf.float32),
  ])
  def concat_zero_dim(self, a, b):
    return tf.concat([a, b], axis=2)

  @tf.function(input_signature=[
      tf.TensorSpec([1, 5, 1], tf.float32),
      tf.TensorSpec([1, 5, 1], tf.float32),
  ])
  def concat0axis(self, a, b):
    return tf.concat([a, b], axis=0)

  @tf.function(input_signature=[
      tf.TensorSpec([1, 5, 1], tf.float32),
      tf.TensorSpec([1, 5, 1], tf.float32),
  ])
  def concat1axis(self, a, b):
    return tf.concat([a, b], axis=1)

  @tf.function(input_signature=[
      tf.TensorSpec([1, 5, 1], tf.float32),
      tf.TensorSpec([1, 5, 1], tf.float32),
  ])
  def concat2axis(self, a, b):
    return tf.concat([a, b], axis=2)


@tf_test_utils.compile_module(ConcatOpsModule)
class ConcatOpsTest(tf_test_utils.SavedModelTestCase):

  def test_concat_zero_dim(self):
    tf_utils.set_random_seed()
    m = self.get_module()
    a = tf.random.uniform([1, 5, 0], dtype=tf.float32)
    b = tf.random.uniform([1, 5, 1], dtype=tf.float32)
    dst = m.concat_zero_dim(a, b)
    dst.assert_all_close()

  def concat0axis(self):
    tf_utils.set_random_seed()
    m = self.get_module()
    a = tf.random.uniform([1, 5, 1], dtype=tf.float32)
    b = tf.random.uniform([1, 5, 1], dtype=tf.float32)
    dst = m.concat_zero_dim(a, b)
    dst.assert_all_close()

  def concat1axis(self):
    tf_utils.set_random_seed()
    m = self.get_module()
    a = tf.random.uniform([1, 5, 1], dtype=tf.float32)
    b = tf.random.uniform([1, 5, 1], dtype=tf.float32)
    dst = m.concat_zero_dim(a, b)
    dst.assert_all_close()

  def concat2axis(self):
    tf_utils.set_random_seed()
    m = self.get_module()
    a = tf.random.uniform([1, 5, 1], dtype=tf.float32)
    b = tf.random.uniform([1, 5, 1], dtype=tf.float32)
    dst = m.concat_zero_dim(a, b)
    dst.assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
