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
import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class FftModule(tf.Module):
  # TODO(natashaknk) when multiple outputs are supported, make into one test.
  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def fft_real(self, real_array, imag_array):
    complex_in = tf.complex(real_array, imag_array)
    complex_out = tf.signal.fft(complex_in)
    return tf.math.real(complex_out)

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def fft_imag(self, real_array, imag_array):
    complex_in = tf.complex(real_array, imag_array)
    complex_out = tf.signal.fft(complex_in)
    return tf.math.imag(complex_out)


class FftTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(FftModule)

  def test_fft_real(self):

    def fft_real(module):
      real_array = np.array([9., 1., 4.5, -0.3], dtype=np.float32)
      imag_array = np.array([0., -1., 17.7, 10.], dtype=np.float32)
      module.fft_real(real_array, imag_array)

    self.compare_backends(fft_real, self._modules)

  def test_fft_imag(self):

    def fft_imag(module):
      real_array = np.array([9., 1., 4.5, -0.3], dtype=np.float32)
      imag_array = np.array([0., -1., 17.7, 10.], dtype=np.float32)
      module.fft_imag(real_array, imag_array)

    self.compare_backends(fft_imag, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
