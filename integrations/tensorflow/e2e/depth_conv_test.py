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
import tensorflow.compat.v2 as tf


class Conv2dModule(tf.Module):

  # TODO(ataei): Add dilation and strided tests.
  @tf.function(input_signature=[
      tf.TensorSpec([2, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 2, 2, 3], tf.float32),
  ])
  def conv2d_2452x2223_valid(self, img, kernel):
    return tf.nn.depthwise_conv2d(
        img, kernel, [1, 1, 1, 1], "VALID", name="result")

  @tf.function(input_signature=[
      tf.TensorSpec([2, 4, 5, 2], tf.float32),
      tf.TensorSpec([2, 4, 2, 3], tf.float32),
  ])
  def conv2d_2452x2223_same(self, img, kernel):
    return tf.nn.depthwise_conv2d(
        img, kernel, [1, 1, 1, 1], "SAME", name="result")


@tf_test_utils.compile_module(Conv2dModule)
class ConvTest(tf_test_utils.SavedModelTestCase):

  def test_batched_feature_unpadded(self):
    i = np.arange(80, dtype=np.float32).reshape([2, 4, 5, 2])
    k = np.arange(24, dtype=np.float32).reshape([2, 2, 2, 3])
    r = self.get_module().conv2d_2452x2223_valid(i, k)
    r.print().assert_all_close()

  def test_batched_feature_unpadded_smae(self):
    i = np.arange(80, dtype=np.float32).reshape([2, 4, 5, 2])
    k = np.arange(48, dtype=np.float32).reshape([2, 4, 2, 3])
    r = self.get_module().conv2d_2452x2223_same(i, k)
    r.print().assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
