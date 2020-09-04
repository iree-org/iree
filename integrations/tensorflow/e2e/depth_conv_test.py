# Lint as: python3
# Copyright 2019 Google LLC
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

import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf


class DepthConv2dModule(tf.Module):

  # TODO(ataei): Add dilation and strided tests.
  @tf.function(input_signature=[
      tf.TensorSpec([2, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 2, 2, 3], tf.float32),
  ])
  def conv2d_2452x2423_valid(self, img, kernel):
    return tf.nn.depthwise_conv2d(
        img, kernel, [1, 1, 1, 1], "VALID", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([2, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 4, 2, 3], tf.float32),
  ])
  def conv2d_2452x2423_same(self, img, kernel):
    return tf.nn.depthwise_conv2d(
        img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([2, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 4, 2, 3], tf.float32),
  ])
  def conv2d_2452x2423_valid_stride_2(self, img, kernel):
    return tf.nn.depthwise_conv2d(
        img, kernel, [1, 2, 2, 1], "VALID", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([2, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 4, 2, 3], tf.float32),
  ])
  def conv2d_2452x2423_same_stride_2(self, img, kernel):
    return tf.nn.depthwise_conv2d(
        img, kernel, [1, 2, 2, 1], "SAME", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([2, 4, 5, 4], tf.float32),
      tf.TensorSpec([2, 4, 4, 1], tf.float32),
  ])
  def conv2d_2453x2441_same_stride_1(self, img, kernel):
    return tf.nn.depthwise_conv2d(
        img, kernel, [1, 1, 1, 1], "SAME", name="result")


@tf_test_utils.compile_module(DepthConv2dModule)
class ConvTest(tf_test_utils.TracedModuleTestCase):

  def test_batched_feature_unpadded(self):

    def batched_feature_unpadded(module):
      i = tf_utils.ndarange([2, 4, 5, 2])
      k = tf_utils.ndarange([2, 2, 2, 3])
      module.conv2d_2452x2423_valid(i, k)

    self.compare_backends(batched_feature_unpadded)

  def test_batched_feature_unpadded_same(self):

    def batched_feature_unpadded_same(module):
      i = tf_utils.ndarange([2, 4, 5, 2])
      k = tf_utils.ndarange([2, 4, 2, 3])
      module.conv2d_2452x2423_same(i, k)

    self.compare_backends(batched_feature_unpadded_same)

  def test_batched_feature_unpadded_same_stride_2(self):

    def batched_feature_unpadded_same_stride_2(module):
      i = tf_utils.ndarange([2, 4, 5, 2])
      k = tf_utils.ndarange([2, 4, 2, 3])
      module.conv2d_2452x2423_valid_stride_2(i, k)

    self.compare_backends(batched_feature_unpadded_same_stride_2)

  def test_batched_feature_padded_same_stride_2(self):

    def batched_feature_padded_same_stride_2(module):
      i = tf_utils.ndarange([2, 4, 5, 2])
      k = tf_utils.ndarange([2, 4, 2, 3])
      module.conv2d_2452x2423_same_stride_2(i, k)

    self.compare_backends(batched_feature_padded_same_stride_2)

  def test_batched_feature_padded_same_stride_1_output_1(self):

    def batched_feature_padded_same_stride_1_output_1(module):
      i = tf_utils.ndarange([2, 4, 5, 4])
      k = tf_utils.ndarange([2, 4, 4, 1])
      module.conv2d_2453x2441_same_stride_1(i, k)

    self.compare_backends(batched_feature_padded_same_stride_1_output_1)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
