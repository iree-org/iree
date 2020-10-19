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

from absl import app
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


class ConvTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(Conv2dModule)

  # yapf: disable
  def test_id_batch_size_1(self):
    def id_batch_size_1(module):
      i = tf_utils.ndarange([1, 4, 5, 1])
      k = np.ones([1, 1, 1, 1], dtype=np.float32)
      module.conv2d_1451x1111_valid(i, k)
    self.compare_backends(id_batch_size_1, self._modules)

  def test_id_batch_size_2(self):
    def id_batch_size_2(module):
      i = tf_utils.ndarange([2, 4, 5, 1])
      k = np.ones([1, 1, 1, 1], dtype=np.float32)
      module.conv2d_2451x1111_valid(i, k)
    self.compare_backends(id_batch_size_2, self._modules)

  def test_asymmetric_kernel(self):
    def asymmetric_kernel(module):
      i = tf_utils.ndarange([1, 4, 5, 1])
      k = np.array([[1, 4, 2], [-2, 0, 1]],
                   dtype=np.float32).reshape(2, 3, 1, 1)
      module.conv2d_1451x2311_valid(i, k)
    self.compare_backends(asymmetric_kernel, self._modules)

  def test_padding(self):
    def padding(module):
      i = tf_utils.ndarange([1, 4, 5, 1])
      k = np.array([[1, 4, 2], [-2, 0, 1]],
                   dtype=np.float32).reshape(2, 3, 1, 1)
      module.conv2d_1451x2311_same(i, k)
    self.compare_backends(padding, self._modules)

  def test_batched_padding(self):
    def batched_padding(module):
      i = tf_utils.ndarange([2, 4, 5, 1])
      k = np.array([[1, 4, 2], [-2, 0, 1]],
                   dtype=np.float32).reshape(2, 3, 1, 1)
      module.conv2d_2451x2311_same(i, k)
    self.compare_backends(batched_padding, self._modules)

  def test_feature_reduce(self):
    def feature_reduce(module):
      i = tf_utils.ndarange([1, 4, 5, 2])
      k = np.ones([3, 2, 2, 1], dtype=np.float32)
      module.conv2d_1452x3221_same(i, k)
    self.compare_backends(feature_reduce, self._modules)

  def test_feature_inflate(self):
    def feature_inflate(module):
      i = tf_utils.ndarange([1, 4, 5, 1])
      k = tf_utils.ndarange([1, 1, 1, 2])
      module.conv2d_1451x1112_same(i, k)
    self.compare_backends(feature_inflate, self._modules)

  def test_feature_mix(self):
    def feature_mix(module):
      i = tf_utils.ndarange([1, 4, 5, 2])
      k = tf_utils.ndarange([1, 1, 2, 2])
      module.conv2d_1452x1122_same(i, k)
    self.compare_backends(feature_mix, self._modules)

  def test_feature_padded(self):
    def feature_padded(module):
      i = tf_utils.ndarange([1, 4, 5, 2])
      k = tf_utils.ndarange([2, 2, 2, 3])
      module.conv2d_1452x2223_same(i, k)
    self.compare_backends(feature_padded, self._modules)

  def test_feature_unpadded(self):
    def feature_unpadded(module):
      i = tf_utils.ndarange([1, 4, 5, 2])
      k = tf_utils.ndarange([2, 2, 2, 3])
      module.conv2d_1452x2223_valid(i, k)
    self.compare_backends(feature_unpadded, self._modules)

  def test_batched_feature_unpadded(self):
    def batched_feature_unpadded(module):
      i = tf_utils.ndarange([2, 4, 5, 2])
      k = tf_utils.ndarange([2, 2, 2, 3])
      module.conv2d_2452x2223_valid(i, k)
    self.compare_backends(batched_feature_unpadded, self._modules)
  # yapf: enable


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
