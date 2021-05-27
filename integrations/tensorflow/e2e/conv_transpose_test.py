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


class ConvTransposeModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([2, 2, 1, 1], tf.float32),
      tf.TensorSpec([1, 2, 4, 1], tf.float32),
  ])
  def conv2d_transpose_same(self, filt, img):
    input_sizes = [1, 2, 4, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"
    return tf.nn.conv2d_transpose(img,
                                  filt,
                                  input_sizes,
                                  strides,
                                  padding,
                                  name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([1, 4, 2, 3], tf.float32),
      tf.TensorSpec([1, 1, 4, 3], tf.float32),
  ])
  def conv2d_transpose_dilated_w(self, filt, img):
    input_sizes = [1, 1, 10, 2]
    strides = [1, 1, 2, 1]
    padding = "VALID"
    return tf.nn.conv2d_transpose(img,
                                  filt,
                                  input_sizes,
                                  strides,
                                  padding,
                                  name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([4, 1, 2, 3], tf.float32),
      tf.TensorSpec([1, 4, 1, 3], tf.float32),
  ])
  def conv2d_transpose_dilated_h(self, filt, img):
    input_sizes = [1, 10, 1, 2]
    strides = [1, 2, 1, 1]
    padding = "VALID"
    return tf.nn.conv2d_transpose(img,
                                  filt,
                                  input_sizes,
                                  strides,
                                  padding,
                                  name="result")


class ConvTransposeTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(ConvTransposeModule)

  # yapf: disable
  def test_transposed(self):
    def transposed(module):
      kernel = tf_utils.uniform([2, 2, 1, 1], dtype=np.float32)
      img = tf_utils.uniform([1, 2, 4, 1], dtype=np.float32)

      module.conv2d_transpose_same(kernel, img)
    self.compare_backends(transposed, self._modules)

  def test_transposed_dilated_w(self):
    def transposed(module):
      kernel = tf_utils.uniform([1, 4, 2, 3], dtype=np.float32)
      img = tf_utils.uniform([1, 1, 4, 3], dtype=np.float32)

      module.conv2d_transpose_dilated_w(kernel, img)
    self.compare_backends(transposed, self._modules)

  def test_transposed_dilated_h(self):
    def transposed(module):
      kernel = tf_utils.uniform([4, 1, 2, 3], dtype=np.float32)
      img = tf_utils.uniform([1, 4, 1, 3], dtype=np.float32)

      module.conv2d_transpose_dilated_h(kernel, img)
    self.compare_backends(transposed, self._modules)
  # yapf: enable


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
