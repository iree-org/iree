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
"""Batch norm tests."""

import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf


class BatchNormModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4, 16], tf.float32),
      tf.TensorSpec([16], tf.float32),
      tf.TensorSpec([16], tf.float32),
      tf.TensorSpec([16], tf.float32),
      tf.TensorSpec([16], tf.float32),
  ])
  def batch_norm_inference(self, x, mean, variance, offset, scale):
    return tf.nn.batch_normalization(
        x,
        mean=mean,
        variance=variance,
        offset=offset,
        scale=scale,
        variance_epsilon=1e-4)


@tf_test_utils.compile_module(BatchNormModule)
class BatchNormTest(tf_test_utils.TracedModuleTestCase):

  def test_batch_norm_inference(self):

    def batch_norm_inference(module):
      # Note: scaling by a small value to increase numerical stability.
      x = tf_utils.uniform((4, 16)) * 1e-3
      mean = tf_utils.uniform((16,)) * 1e-3
      variance = tf_utils.uniform((16,)) * 1e-3
      offset = tf_utils.uniform((16,)) * 1e-3
      scale = tf_utils.uniform((16,)) * 1e-3
      module.batch_norm_inference(x, mean, variance, offset, scale)

    self.compare_backends(batch_norm_inference)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
