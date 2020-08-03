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

import numpy as np
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
class ConcatOpsTest(tf_test_utils.TracedModuleTestCase):

  def test_concat_zero_dim(self):

    def concat_zero_dim(module):
      a = tf_utils.uniform([1, 5, 0])
      b = tf_utils.uniform([1, 5, 1])
      module.concat_zero_dim(a, b)

    self.compare_backends(concat_zero_dim)

  def test_concat0axis(self):

    def concat0axis(module):
      a = tf_utils.uniform([1, 5, 1])
      b = tf_utils.uniform([1, 5, 1])
      module.concat0axis(a, b)

    self.compare_backends(concat0axis)

  def test_concat1axis(self):

    def concat1axis(module):
      a = tf_utils.uniform([1, 5, 1])
      b = tf_utils.uniform([1, 5, 1])
      module.concat1axis(a, b)

    self.compare_backends(concat1axis)

  def test_concat2axis(self):

    def concat2axis(module):
      a = tf_utils.uniform([1, 5, 1])
      b = tf_utils.uniform([1, 5, 1])
      module.concat2axis(a, b)

    self.compare_backends(concat2axis)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
