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


class Conv2dModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([1, 4, 5, 1], tf.float32),
      tf.TensorSpec([1, 1, 1, 1], tf.float32),
  ])
  def conv2d_1451x1111_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([2, 4, 5, 1], tf.float32),
      tf.TensorSpec([1, 1, 1, 1], tf.float32),
  ])
  def conv2d_2451x1111_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([1, 4, 5, 1], tf.float32),
      tf.TensorSpec([2, 3, 1, 1], tf.float32),
  ])
  def conv2d_1451x2311_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([1, 4, 5, 1], tf.float32),
      tf.TensorSpec([2, 3, 1, 1], tf.float32),
  ])
  def conv2d_1451x2311_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([2, 4, 5, 1], tf.float32),
      tf.TensorSpec([2, 3, 1, 1], tf.float32),
  ])
  def conv2d_2451x2311_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([1, 4, 5, 2], tf.float32),
      tf.TensorSpec([3, 2, 2, 1], tf.float32),
  ])
  def conv2d_1452x3221_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([1, 4, 5, 1], tf.float32),
      tf.TensorSpec([1, 1, 1, 2], tf.float32),
  ])
  def conv2d_1451x1112_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([1, 4, 5, 2], tf.float32),
      tf.TensorSpec([1, 1, 2, 2], tf.float32),
  ])
  def conv2d_1452x1122_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([1, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 2, 2, 3], tf.float32),
  ])
  def conv2d_1452x2223_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([1, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 2, 2, 3], tf.float32),
  ])
  def conv2d_1452x2223_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([2, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 2, 2, 3], tf.float32),
  ])
  def conv2d_2452x2223_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")


@tf_test_utils.compile_module(Conv2dModule)
class ConvTest(tf_test_utils.TracedModuleTestCase):

  def test_id_batch_size_1(self):

    def id_batch_size_1(module):
      i = tf_utils.ndarange([1, 4, 5, 1])
      k = np.ones([1, 1, 1, 1], dtype=np.float32)
      module.conv2d_1451x1111_valid(i, k)

    self.compare_backends(id_batch_size_1)

  def test_id_batch_size_2(self):

    def id_batch_size_2(module):
      i = tf_utils.ndarange([2, 4, 5, 1])
      k = np.ones([1, 1, 1, 1], dtype=np.float32)
      module.conv2d_2451x1111_valid(i, k)

    self.compare_backends(id_batch_size_2)

  def test_asymmetric_kernel(self):

    def asymmetric_kernel(module):
      i = tf_utils.ndarange([1, 4, 5, 1])
      k = np.array([[1, 4, 2], [-2, 0, 1]],
                   dtype=np.float32).reshape(2, 3, 1, 1)
      module.conv2d_1451x2311_valid(i, k)

    self.compare_backends(asymmetric_kernel)

  def test_padding(self):

    def padding(module):
      i = tf_utils.ndarange([1, 4, 5, 1])
      k = np.array([[1, 4, 2], [-2, 0, 1]],
                   dtype=np.float32).reshape(2, 3, 1, 1)
      module.conv2d_1451x2311_same(i, k)

    self.compare_backends(padding)

  def test_batched_padding(self):

    def batched_padding(module):
      i = tf_utils.ndarange([2, 4, 5, 1])
      k = np.array([[1, 4, 2], [-2, 0, 1]],
                   dtype=np.float32).reshape(2, 3, 1, 1)
      module.conv2d_2451x2311_same(i, k)

    self.compare_backends(batched_padding)

  def test_feature_reduce(self):

    def feature_reduce(module):
      i = tf_utils.ndarange([1, 4, 5, 2])
      k = np.ones([3, 2, 2, 1], dtype=np.float32)
      module.conv2d_1452x3221_same(i, k)

    self.compare_backends(feature_reduce)

  def test_feature_inflate(self):

    def feature_inflate(module):
      i = tf_utils.ndarange([1, 4, 5, 1])
      k = tf_utils.ndarange([1, 1, 1, 2])
      module.conv2d_1451x1112_same(i, k)

    self.compare_backends(feature_inflate)

  def test_feature_mix(self):

    def feature_mix(module):
      i = tf_utils.ndarange([1, 4, 5, 2])
      k = tf_utils.ndarange([1, 1, 2, 2])
      module.conv2d_1452x1122_same(i, k)

    self.compare_backends(feature_mix)

  def test_feature_padded(self):

    def feature_padded(module):
      i = tf_utils.ndarange([1, 4, 5, 2])
      k = tf_utils.ndarange([2, 2, 2, 3])
      module.conv2d_1452x2223_same(i, k)

    self.compare_backends(feature_padded)

  def test_feature_unpadded(self):

    def feature_unpadded(module):
      i = tf_utils.ndarange([1, 4, 5, 2])
      k = tf_utils.ndarange([2, 2, 2, 3])
      module.conv2d_1452x2223_valid(i, k)

    self.compare_backends(feature_unpadded)

  def test_batched_feature_unpadded(self):

    def batched_feature_unpadded(module):
      i = tf_utils.ndarange([2, 4, 5, 2])
      k = tf_utils.ndarange([2, 2, 2, 3])
      module.conv2d_2452x2223_valid(i, k)

    self.compare_backends(batched_feature_unpadded)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
