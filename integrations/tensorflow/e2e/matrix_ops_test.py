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
"""Test matrix ops."""

from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class MatrixOpsModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4, 2], tf.float32),
      tf.TensorSpec([2, 4], tf.float32),
  ])
  def basic_matmul(self, lhs, rhs):
    return tf.matmul(lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([3, 4, 2], tf.float32),
      tf.TensorSpec([2, 4], tf.float32),
  ])
  def matmul_lhs_batch(self, lhs, rhs):
    return tf.matmul(lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 2], tf.float32),
      tf.TensorSpec([3, 2, 4], tf.float32),
  ])
  def matmul_rhs_batch(self, lhs, rhs):
    return tf.matmul(lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([1, 4, 2], tf.float32),
      tf.TensorSpec([3, 2, 4], tf.float32),
  ])
  def matmul_broadcast_singleton_dimension(self, lhs, rhs):
    return tf.matmul(lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([None, None, 4, 2], tf.float32),
      tf.TensorSpec([None, None, 2, 4], tf.float32),
  ])
  def matmul_high_rank_batch(self, lhs, rhs):
    return tf.matmul(lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([None, None, None], tf.float32),
      tf.TensorSpec([None, None, None], tf.float32),
  ])
  def matmul_dynamic(self, lhs, rhs):
    return tf.matmul(lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([None, None, None], tf.float32),
      tf.TensorSpec([None, None], tf.float32),
  ])
  def matmul_dynamic_lhs_batch(self, lhs, rhs):
    return tf.matmul(lhs, rhs)


@tf_test_utils.compile_module(MatrixOpsModule)
class MatrixOpsTest(tf_test_utils.SavedModelTestCase):

  def test_basic_matmul(self):
    m = self.get_module()
    dst = m.basic_matmul(tf.random.uniform([4, 2]), tf.random.uniform([2, 4]))
    dst.assert_all_close()

  def test_matmul_lhs_batch(self):
    m = self.get_module()
    dst = m.matmul_lhs_batch(
        tf.random.uniform([3, 4, 2]), tf.random.uniform([2, 4]))
    dst.assert_all_close()

  def test_matmul_rhs_batch(self):
    m = self.get_module()
    dst = m.matmul_rhs_batch(
        tf.random.uniform([4, 2]), tf.random.uniform([3, 2, 4]))
    dst.assert_all_close()

  def test_matmul_broadcast_singleton_dimension(self):
    m = self.get_module()
    dst = m.matmul_broadcast_singleton_dimension(
        tf.random.uniform([1, 4, 2]), tf.random.uniform([3, 2, 4]))
    dst.assert_all_close()

  def test_matmul_high_rank_batch(self):
    m = self.get_module()
    dst = m.matmul_high_rank_batch(
        tf.random.uniform([1, 7, 4, 2]), tf.random.uniform([7, 1, 2, 4]))
    dst.assert_all_close()

  def test_matmul_dynamic_matching_batch(self):
    m = self.get_module()
    dst = m.matmul_dynamic(
        tf.random.uniform([2, 2, 3]), tf.random.uniform([2, 3, 4]))
    dst.assert_all_close()

  def test_matmul_dynamic_broadcast_lhs(self):
    m = self.get_module()
    dst = m.matmul_dynamic(
        tf.random.uniform([1, 2, 3]), tf.random.uniform([2, 3, 4]))
    dst.assert_all_close()

  def test_matmul_dynamic_broadcast_rhs(self):
    m = self.get_module()
    dst = m.matmul_dynamic(
        tf.random.uniform([2, 2, 3]), tf.random.uniform([1, 3, 4]))
    dst.assert_all_close()

  def test_matmul_dynamic_rank_broadcasting(self):
    m = self.get_module()
    dst = m.matmul_dynamic_lhs_batch(
        tf.random.uniform([7, 2, 3]), tf.random.uniform([3, 4]))
    dst.assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
