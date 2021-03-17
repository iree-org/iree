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

from absl import app
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

LEFT_DIM = 64
INNER_DIM = 32
RIGHT_DIM = 16
BATCH_DIM = 256


class MatrixOpsStaticModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, INNER_DIM], tf.float32),
      tf.TensorSpec([INNER_DIM, RIGHT_DIM], tf.float32),
  ])
  def basic_matmul(self, lhs, rhs):
    return tf.matmul(lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_DIM, LEFT_DIM, INNER_DIM], tf.float32),
      tf.TensorSpec([INNER_DIM, RIGHT_DIM], tf.float32),
  ])
  def matmul_lhs_batch(self, lhs, rhs):
    return tf.matmul(lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, INNER_DIM], tf.float32),
      tf.TensorSpec([BATCH_DIM, INNER_DIM, RIGHT_DIM], tf.float32),
  ])
  def matmul_rhs_batch(self, lhs, rhs):
    return tf.matmul(lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([1, LEFT_DIM, INNER_DIM], tf.float32),
      tf.TensorSpec([BATCH_DIM, INNER_DIM, RIGHT_DIM], tf.float32),
  ])
  def matmul_broadcast_singleton_dimension(self, lhs, rhs):
    return tf.matmul(lhs, rhs)


class MatrixOpsStaticTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(MatrixOpsStaticModule)

  # yapf: disable
  def test_basic_matmul(self):
    def basic_matmul(module):
      module.basic_matmul(tf_utils.uniform([LEFT_DIM, INNER_DIM]),
                          tf_utils.uniform([INNER_DIM, RIGHT_DIM]))
    self.compare_backends(basic_matmul, self._modules)

  def test_matmul_lhs_batch(self):
    def matmul_lhs_batch(module):
      module.matmul_lhs_batch(
          tf_utils.uniform([BATCH_DIM, LEFT_DIM, INNER_DIM]),
          tf_utils.uniform([INNER_DIM, RIGHT_DIM]))
    self.compare_backends(matmul_lhs_batch, self._modules)

  def test_matmul_rhs_batch(self):
    def matmul_rhs_batch(module):
      module.matmul_rhs_batch(
          tf_utils.uniform([LEFT_DIM, INNER_DIM]),
          tf_utils.uniform([BATCH_DIM, INNER_DIM, RIGHT_DIM]))
    self.compare_backends(matmul_rhs_batch, self._modules)

  def test_matmul_broadcast_singleton_dimension(self):
    def matmul_broadcast_singleton_dimension(module):
      module.matmul_broadcast_singleton_dimension(
          tf_utils.uniform([1, LEFT_DIM, INNER_DIM]),
          tf_utils.uniform([BATCH_DIM, INNER_DIM, RIGHT_DIM]))
    self.compare_backends(matmul_broadcast_singleton_dimension, self._modules)
  # yapf: enable


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
