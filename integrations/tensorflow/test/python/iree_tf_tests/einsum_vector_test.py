# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test matrix ops via einsum"""

from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

VECTOR_DIM = 16


class EinsumVectorModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([VECTOR_DIM], tf.float32),
  ])
  def einsum_identity(self, x):
    return tf.einsum('i', x)

  @tf.function(input_signature=[
      tf.TensorSpec([VECTOR_DIM], tf.float32),
  ])
  def einsum_sum(self, x):
    return tf.einsum('i ->', x)

  @tf.function(input_signature=[
      tf.TensorSpec([VECTOR_DIM], tf.float32),
      tf.TensorSpec([VECTOR_DIM], tf.float32),
  ])
  def einsum_mul(self, lhs, rhs):
    return tf.einsum('i, i -> i', lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([VECTOR_DIM], tf.float32),
      tf.TensorSpec([VECTOR_DIM], tf.float32),
  ])
  def einsum_implicit_inner_product(self, lhs, rhs):
    return tf.einsum('i, i', lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([VECTOR_DIM], tf.float32),
      tf.TensorSpec([VECTOR_DIM], tf.float32),
  ])
  def einsum_explicit_inner_product(self, lhs, rhs):
    return tf.einsum('i, i ->', lhs, rhs)

  @tf.function(input_signature=[
      tf.TensorSpec([VECTOR_DIM], tf.float32),
      tf.TensorSpec([VECTOR_DIM], tf.float32),
  ])
  def einsum_outer_product(self, lhs, rhs):
    return tf.einsum('i, j -> ij', lhs, rhs)


class EinsumVectorTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(EinsumVectorModule)

  # yapf: disable
  def test_einsum_identity(self):
    def einsum_identity(module):
      module.einsum_identity(tf_utils.ndarange([VECTOR_DIM]))
    self.compare_backends(einsum_identity, self._modules)

  def test_einsum_sum(self):
    def einsum_sum(module):
      module.einsum_sum(tf_utils.ndarange([VECTOR_DIM]))
    self.compare_backends(einsum_sum, self._modules)

  def test_einsum_mul(self):
    def einsum_mul(module):
      module.einsum_mul(tf_utils.ndarange([VECTOR_DIM]),
                        tf_utils.ndarange([VECTOR_DIM]))
    self.compare_backends(einsum_mul, self._modules)

  def test_einsum_implicit_inner_product(self):
    def einsum_implicit_inner_product(module):
      module.einsum_implicit_inner_product(tf_utils.ndarange([VECTOR_DIM]),
                                           tf_utils.ndarange([VECTOR_DIM]))
    self.compare_backends(einsum_implicit_inner_product, self._modules)

  def test_einsum_explicit_inner_product(self):
    def einsum_explicit_inner_product(module):
      module.einsum_explicit_inner_product(tf_utils.ndarange([VECTOR_DIM]),
                                           tf_utils.ndarange([VECTOR_DIM]))
    self.compare_backends(einsum_explicit_inner_product, self._modules)

  def test_einsum_outer_product(self):
    def einsum_outer_product(module):
      module.einsum_outer_product(tf_utils.ndarange([VECTOR_DIM]),
                                  tf_utils.ndarange([VECTOR_DIM]))
    self.compare_backends(einsum_outer_product, self._modules)
  # yapf: enable


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
