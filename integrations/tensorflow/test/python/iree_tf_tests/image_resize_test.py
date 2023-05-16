# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
