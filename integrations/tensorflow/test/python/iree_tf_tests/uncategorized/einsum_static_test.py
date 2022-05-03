# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test matrix ops via einsum"""

from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

LEFT_DIM = 6
INNER_DIM = 3
RIGHT_DIM = 6
BATCH_DIM = 8


class EinsumStaticModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_identity(self, x):
    return tf.einsum('ij', x)

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_implicit_transpose(self, x):
    return tf.einsum('ji', x)  # :woozy:

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_explicit_transpose(self, x):
    return tf.einsum('ij -> ji', x)

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_implicit_trace(self, x):
    return tf.einsum('ii', x)

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_explicit_trace(self, x):
    return tf.einsum('ii ->', x)

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_diag(self, x):
    return tf.einsum('ii -> i', x)

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_sum(self, x):
    return tf.einsum('ij ->', x)

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_sum_axis_0(self, x):
    return tf.einsum('ij -> j', x)

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_sum_axis_1(self, x):
    return tf.einsum('ij -> i', x)

  @tf.function(input_signature=[
      tf.TensorSpec([LEFT_DIM, INNER_DIM], tf.float32),
      tf.TensorSpec([INNER_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_matmul(self, lhs, rhs):
    return tf.einsum('ij, jk -> ik', lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_DIM, LEFT_DIM, INNER_DIM], tf.float32),
      tf.TensorSpec([INNER_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_lhs_batch(self, lhs, rhs):
    return tf.einsum('bij, jk -> bik', lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([1, LEFT_DIM, INNER_DIM], tf.float32),
      tf.TensorSpec([BATCH_DIM, INNER_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_broadcast_singleton_dimension(self, lhs, rhs):
    return tf.einsum('lij, rjk -> rik', lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_DIM, 8, 6], tf.float32),
      tf.TensorSpec([12, 6, 4], tf.float32),
  ])
  def einsum_split_heads(self, seq, weights):
    # l: seq_len, m: d_model, h: num_heads, d: attention_depth
    return tf.einsum('blm, hmd -> bhld', seq, weights)

  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_DIM, 5, 3, 2, 6], tf.float32),
      tf.TensorSpec([BATCH_DIM, 5, 6], tf.float32),
  ])
  def einsum_batched_high_rank_matrix_vector_mul(self, lhs, rhs):
    return tf.einsum('bijxy, biy -> bijx', lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_DIM, 2, 6], tf.float32),
      tf.TensorSpec([BATCH_DIM, 5, 3, 6], tf.float32),
  ])
  def einsum_batched_matrix_high_rank_vector_mul(self, lhs, rhs):
    return tf.einsum('bxy, bijy -> bijx', lhs, rhs)


class EinsumStaticTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(EinsumStaticModule)

  # yapf: disable
  def test_einsum_identity(self):
    def einsum_identity(module):
      module.einsum_identity(tf_utils.ndarange([LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_identity, self._modules)

  def test_einsum_implicit_transpose(self):
    def einsum_implicit_transpose(module):
      module.einsum_implicit_transpose(tf_utils.ndarange([LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_implicit_transpose, self._modules)

  def test_einsum_explicit_transpose(self):
    def einsum_explicit_transpose(module):
      module.einsum_explicit_transpose(tf_utils.ndarange([LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_explicit_transpose, self._modules)

  def test_einsum_implicit_trace(self):
    def einsum_implicit_trace(module):
      module.einsum_implicit_trace(tf_utils.ndarange([LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_implicit_trace, self._modules)

  def test_einsum_explicit_trace(self):
    def einsum_explicit_trace(module):
      module.einsum_explicit_trace(tf_utils.ndarange([LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_explicit_trace, self._modules)

  def test_einsum_diag(self):
    def einsum_diag(module):
      module.einsum_diag(tf_utils.ndarange([LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_diag, self._modules)

  def test_einsum_sum(self):
    def einsum_sum(module):
      module.einsum_sum(tf_utils.ndarange([LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_sum, self._modules)

  def test_einsum_sum_axis_0(self):
    def einsum_sum_axis_0(module):
      module.einsum_sum_axis_0(tf_utils.ndarange([LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_sum_axis_0, self._modules)

  def test_einsum_sum_axis_1(self):
    def einsum_sum_axis_1(module):
      module.einsum_sum_axis_1(tf_utils.ndarange([LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_sum_axis_1, self._modules)

  def test_einsum_matmul(self):
    def einsum_matmul(module):
      module.einsum_matmul(tf_utils.ndarange([LEFT_DIM, INNER_DIM]),
                           tf_utils.ndarange([INNER_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_matmul, self._modules)

  def test_einsum_lhs_batch(self):
    def einsum_lhs_batch(module):
      module.einsum_lhs_batch(
          tf_utils.ndarange([BATCH_DIM, LEFT_DIM, INNER_DIM]),
          tf_utils.ndarange([INNER_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_lhs_batch, self._modules)

  def test_einsum_broadcast_singleton_dimension(self):
    def einsum_broadcast_singleton_dimension(module):
      module.einsum_broadcast_singleton_dimension(
          tf_utils.ndarange([1, LEFT_DIM, INNER_DIM]),
          tf_utils.ndarange([BATCH_DIM, INNER_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_broadcast_singleton_dimension, self._modules)

  def test_einsum_split_heads(self):
    def einsum_split_heads(module):
      module.einsum_split_heads(tf_utils.ndarange([BATCH_DIM, 8, 6]),
                                tf_utils.ndarange([12, 6, 4]))
    self.compare_backends(einsum_split_heads, self._modules)

  def test_einsum_batched_high_rank_matrix_vector_mul(self):
    def einsum_batched_high_rank_matrix_vector_mul(module):
      module.einsum_batched_high_rank_matrix_vector_mul(
          tf_utils.ndarange([BATCH_DIM, 5, 3, 2, 6]),
          tf_utils.ndarange([BATCH_DIM, 5, 6]))
    self.compare_backends(einsum_batched_high_rank_matrix_vector_mul,
                          self._modules)

  def test_einsum_batched_matrix_high_rank_vector_mul(self):
    def einsum_batched_matrix_high_rank_vector_mul(module):
      module.einsum_batched_matrix_high_rank_vector_mul(
          tf_utils.ndarange([BATCH_DIM, 2, 6]),
          tf_utils.ndarange([BATCH_DIM, 5, 3, 6]))
    self.compare_backends(einsum_batched_matrix_high_rank_vector_mul,
                          self._modules)
  # yapf: enable


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
