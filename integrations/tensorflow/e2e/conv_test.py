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
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf


class Conv2dModule(tf_test_utils.TestModule):

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([1, 4, 5, 1], tf.float32),
      tf.TensorSpec([1, 1, 1, 1], tf.float32),
  ])
  def conv2d_1451x1111_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([1, 4, 5, 1], tf.float32),
      tf.TensorSpec([2, 2, 1, 1], tf.float32),
  ])
  def conv2d_1451x2211_dilated_valid(self, img, kernel):
    return tf.nn.conv2d(img,
                        kernel, [1, 1, 1, 1],
                        "VALID",
                        dilations=[1, 2, 1, 1],
                        name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([1, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 2, 2, 3], tf.float32),
  ])
  def conv2d_1452x2223_dilated_valid(self, img, kernel):
    return tf.nn.conv2d(img,
                        kernel, [1, 1, 1, 1],
                        "VALID",
                        dilations=[1, 2, 1, 1],
                        name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([2, 4, 5, 1], tf.float32),
      tf.TensorSpec([1, 1, 1, 1], tf.float32),
  ])
  def conv2d_2451x1111_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([1, 4, 5, 1], tf.float32),
      tf.TensorSpec([2, 3, 1, 1], tf.float32),
  ])
  def conv2d_1451x2311_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([1, 4, 5, 1], tf.float32),
      tf.TensorSpec([2, 3, 1, 1], tf.float32),
  ])
  def conv2d_1451x2311_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([2, 4, 5, 1], tf.float32),
      tf.TensorSpec([2, 3, 1, 1], tf.float32),
  ])
  def conv2d_2451x2311_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([1, 4, 5, 2], tf.float32),
      tf.TensorSpec([3, 2, 2, 1], tf.float32),
  ])
  def conv2d_1452x3221_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([1, 4, 5, 1], tf.float32),
      tf.TensorSpec([1, 1, 1, 2], tf.float32),
  ])
  def conv2d_1451x1112_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([1, 4, 5, 2], tf.float32),
      tf.TensorSpec([1, 1, 2, 2], tf.float32),
  ])
  def conv2d_1452x1122_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([1, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 2, 2, 3], tf.float32),
  ])
  def conv2d_1452x2223_same(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "SAME", name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([1, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 2, 2, 3], tf.float32),
  ])
  def conv2d_1452x2223_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([2, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 2, 2, 3], tf.float32),
  ])
  def conv2d_2452x2223_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")


class ConvTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(Conv2dModule)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  ConvTest.generate_unit_tests(Conv2dModule)
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
