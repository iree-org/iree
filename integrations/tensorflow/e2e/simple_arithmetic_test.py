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
"""Several baseline e2e simple arithmetic tests."""

import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf


class SimpleArithmeticModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    return a * b

  @tf.function(input_signature=[
      tf.TensorSpec([128, 3072], tf.float32),
      tf.TensorSpec([3072, 256], tf.float32),
  ])
  def simple_matmul(self, a, b):
    return tf.matmul(a, b)


@tf_test_utils.compile_module(SimpleArithmeticModule)
class SimpleArithmeticTest(tf_test_utils.TracedModuleTestCase):

  def test_simple_mul(self):

    def simple_mul(module):
      a = np.array([1., 2., 3., 4.], dtype=np.float32)
      b = np.array([400., 5., 6., 7.], dtype=np.float32)
      c = module.simple_mul(a, b)
      module.simple_mul(a, c)

    self.compare_backends(simple_mul)

  def test_simple_matmul(self):

    def simple_matmul(module):
      # Note: scaling by a small value to increase numerical stability.
      a = tf_utils.uniform((128, 3072)) * 1e-3
      b = tf_utils.uniform((3072, 256)) * 1e-3
      module.simple_matmul(a, b)

    self.compare_backends(simple_matmul)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
