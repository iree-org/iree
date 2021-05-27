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

from absl import app
from iree.tf.support import tf_utils
from iree.tf.support import tf_test_utils
import numpy as np
import tensorflow.compat.v1 as tf


class ResizeImageModule(tf.Module):

  def __init__(self):
    pass

  @tf.function(input_signature=[tf.TensorSpec([1, 52, 37, 1], tf.int32)])
  def downsample_nearest_neighbor(self, image):
    size = np.asarray([8, 7], dtype=np.int32)
    return tf.image.resize_nearest_neighbor(image, size)

  @tf.function(input_signature=[tf.TensorSpec([1, 8, 7, 1], tf.int32)])
  def upsample_nearest_neighbor(self, image):
    size = np.asarray([52, 37], dtype=np.int32)
    return tf.image.resize_nearest_neighbor(image, size)


class ResizeImageTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(ResizeImageModule)

  def test_downsample_nearest_neighbor(self):

    def downsample_nearest_neighbor(module):
      img = tf_utils.ndarange([1, 52, 37, 1], dtype=np.int32)
      module.downsample_nearest_neighbor(img)

    self.compare_backends(downsample_nearest_neighbor, self._modules)

  def test_upsample_nearest_neighbor(self):

    def upsample_nearest_neighbor(module):
      img = tf_utils.ndarange([1, 8, 7, 1], dtype=np.int32)
      module.upsample_nearest_neighbor(img)

    self.compare_backends(upsample_nearest_neighbor, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
