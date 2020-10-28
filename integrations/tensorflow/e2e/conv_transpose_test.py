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


class ConvTransposeModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([1, 16, 16, 32], tf.float32),
      tf.TensorSpec([1, 1, 32, 32], tf.float32),
  ])
  def conv2d_transpose_same(self, filt, img):
    input_sizes = [1, 1, 264, 16]
    strides = [1, 1, 8, 1]
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
      kernel = tf_utils.ndarange([1, 16, 16, 32])
      img = tf_utils.ndarange([1, 1, 32, 32])

      module.conv2d_transpose_same(kernel, img)
    self.compare_backends(transposed, self._modules)
  # yapf: enable


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
