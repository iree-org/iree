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


class EinsumDynamicModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([None, None], tf.float32),
  ])
  def einsum_dynamic_dim_identity(self, x):
    return tf.einsum('ij', x)

  @tf.function(input_signature=[
      tf.TensorSpec([None, None, None], tf.float32),
  ])
  def einsum_dynamic_rank_identity(self, x):
    return tf.einsum('...', x)

  @tf.function(input_signature=[
      tf.TensorSpec([None, LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_dynamic_dim_transpose(self, x):
    return tf.einsum('bij -> bji', x)

  @tf.function(input_signature=[
      tf.TensorSpec([None, None, LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_dynamic_rank_diag(self, x):
    return tf.einsum('...ii -> ...i', x)

  @tf.function(input_signature=[
      tf.TensorSpec([None, None, LEFT_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_dynamic_dim_sum(self, x):
    return tf.einsum('abij -> ab', x)

  @tf.function(input_signature=[
      tf.TensorSpec([None, None], tf.float32),
      tf.TensorSpec([None, None], tf.float32),
  ])
  def einsum_dynamic_dim_matmul(self, lhs, rhs):
    return tf.einsum('ij, jk -> ik', lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([None, LEFT_DIM, INNER_DIM], tf.float32),
      tf.TensorSpec([INNER_DIM, RIGHT_DIM], tf.float32),
  ])
  def einsum_dynamic_dim_lhs_batch(self, lhs, rhs):
    return tf.einsum('bij, jk -> bik', lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([None, None, 8, 6], tf.float32),
      tf.TensorSpec([12, 6, 4], tf.float32),
  ])
  def einsum_dynamic_rank_split_heads(self, seq, weights):
    # l: seq_len, m: d_model, h: num_heads, d: attention_depth
    return tf.einsum('...lm, hmd -> ...hld', seq, weights)


class EinsumDynamicTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(EinsumDynamicModule)

  # yapf: disable
  def test_einsum_dynamic_dim_identity(self):
    def einsum_dynamic_dim_identity(module):
      module.einsum_dynamic_dim_identity(
          tf_utils.ndarange([LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_dynamic_dim_identity, self._modules)

  def test_einsum_dynamic_rank_identity(self):
    def einsum_dynamic_rank_identity(module):
      module.einsum_dynamic_rank_identity(
          tf_utils.ndarange([BATCH_DIM, LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_dynamic_rank_identity, self._modules)

  def test_einsum_dynamic_dim_transpose(self):
    def einsum_dynamic_dim_transpose(module):
      module.einsum_dynamic_dim_transpose(
          tf_utils.ndarange([BATCH_DIM, LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_dynamic_dim_transpose, self._modules)

  def test_einsum_dynamic_rank_diag(self):
    def einsum_dynamic_rank_diag(module):
      module.einsum_dynamic_rank_diag(
          tf_utils.ndarange([BATCH_DIM, BATCH_DIM, LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_dynamic_rank_diag, self._modules)

  def test_einsum_dynamic_dim_sum(self):
    def einsum_dynamic_dim_sum(module):
      module.einsum_dynamic_dim_sum(
           tf_utils.ndarange([BATCH_DIM, BATCH_DIM, LEFT_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_dynamic_dim_sum, self._modules)

  def test_einsum_dynamic_dim_matmul(self):
    def einsum_dynamic_dim_matmul(module):
      module.einsum_dynamic_dim_matmul(
          tf_utils.ndarange([LEFT_DIM, INNER_DIM]),
          tf_utils.ndarange([INNER_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_dynamic_dim_matmul, self._modules)

  def test_einsum_dynamic_dim_lhs_batch(self):
    def einsum_dynamic_dim_lhs_batch(module):
      module.einsum_dynamic_dim_lhs_batch(
          tf_utils.ndarange([BATCH_DIM, LEFT_DIM, INNER_DIM]),
          tf_utils.ndarange([INNER_DIM, RIGHT_DIM]))
    self.compare_backends(einsum_dynamic_dim_lhs_batch, self._modules)

  def test_einsum_dynamic_rank_split_heads(self):
    def einsum_dynamic_rank_split_heads(module):
      module.einsum_dynamic_rank_split_heads(
          tf_utils.ndarange([BATCH_DIM, BATCH_DIM, 8, 6]),
          tf_utils.ndarange([12, 6, 4]))
    self.compare_backends(einsum_dynamic_rank_split_heads, self._modules)
  # yapf: enable


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
