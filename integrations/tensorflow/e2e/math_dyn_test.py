# Lint as: python3
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
"""Tests for ops in the tf.math module."""

from absl import app
import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class MathModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
  def abs(self, x):
    return tf.math.abs(x)

  @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
  def ceil(self, x):
    return tf.math.ceil(x)

  @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
  def cos(self, x):
    return tf.math.cos(x)

  @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
  def log(self, x):
    return tf.math.log(x)

  @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
  def mod(self, x):
    return tf.math.mod(x, 2.0)

  @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
  def fake_quant(self, x):
    return tf.quantization.fake_quant_with_min_max_args(x,
                                                        min=-6,
                                                        max=6,
                                                        num_bits=8,
                                                        narrow_range=False,
                                                        name=None)


class MathTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(MathModule)

  # yapf: disable
  def test_abs(self):
    def abs(module):
      module.abs(np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32))
    self.compare_backends(abs, self._modules)

  def test_ceil(self):
    def ceil(module):
      module.ceil(np.array([0.0, 1.2, 1.5, 3.75], dtype=np.float32))
    self.compare_backends(ceil, self._modules)

  def test_cos(self):
    def cos(module):
      module.cos(np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32))
    self.compare_backends(cos, self._modules)

  def test_log(self):
    def log(module):
      module.log(np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float32))
    self.compare_backends(log, self._modules)

  def test_mod(self):
    def mod(module):
      module.mod(np.array([0.0, 1.2, 1.5, 3.75], dtype=np.float32))
    self.compare_backends(mod, self._modules)

  def test_fake_quant(self):
    def abs(module):
      module.fake_quant(np.array([-0.123, 0.1234, 0.743, 4.3], dtype=np.float32))
    self.compare_backends(abs, self._modules)
  # yapf: enable


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
