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

from absl import app
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
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


class SimpleArithmeticTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(SimpleArithmeticModule)

  def test_simple_mul(self):

    def simple_mul(module):
      a = np.array([1., 2., 3., 4.], dtype=np.float32)
      b = np.array([400., 5., 6., 7.], dtype=np.float32)
      c = module.simple_mul(a, b)
      module.simple_mul(a, c)

    self.compare_backends(simple_mul, self._modules)

  def test_simple_matmul(self):

    def simple_matmul(module):
      # Note: scaling by a small value to increase numerical stability.
      a = tf_utils.uniform((128, 3072)) * 1e-3
      b = tf_utils.uniform((3072, 256)) * 1e-3
      module.simple_matmul(a, b)

    self.compare_backends(simple_matmul, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
