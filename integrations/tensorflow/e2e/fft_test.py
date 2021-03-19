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
from iree.tf.support import tf_test_utils
import numpy as np
import tensorflow.compat.v2 as tf


class FftModule(tf.Module):
  # TODO(natashaknk) when multiple outputs are supported, make into one test.

  complex_input_signature = [
      tf.TensorSpec([16], tf.float32),
      tf.TensorSpec([16], tf.float32)
  ]

  real_input_signature = [tf.TensorSpec([32], tf.float32)]

  @tf.function(input_signature=complex_input_signature)
  def fft_real(self, real_array, imag_array):
    complex_in = tf.complex(real_array, imag_array)
    complex_out = tf.signal.fft(complex_in)
    return tf.math.real(complex_out)

  @tf.function(input_signature=complex_input_signature)
  def fft_imag(self, real_array, imag_array):
    complex_in = tf.complex(real_array, imag_array)
    complex_out = tf.signal.fft(complex_in)
    return tf.math.imag(complex_out)

  @tf.function(input_signature=complex_input_signature)
  def ifft_real(self, real_array, imag_array):
    complex_in = tf.complex(real_array, imag_array)
    complex_out = tf.signal.ifft(complex_in)
    return tf.math.real(complex_out)

  @tf.function(input_signature=complex_input_signature)
  def ifft_imag(self, real_array, imag_array):
    complex_in = tf.complex(real_array, imag_array)
    complex_out = tf.signal.ifft(complex_in)
    return tf.math.imag(complex_out)

  @tf.function(input_signature=real_input_signature)
  def rfft_real(self, real_array):
    complex_out = tf.signal.rfft(real_array)
    return tf.math.real(complex_out)

  @tf.function(input_signature=real_input_signature)
  def rfft_imag(self, real_array):
    complex_out = tf.signal.rfft(real_array)
    return tf.math.imag(complex_out)

  # TODO(natashaknk): Enable IRFFT tests when Linalg on tensors changes land.
  # @tf.function(input_signature=complex_input_signature)
  # def irfft(self, real_array, imag_array):
  #   complex_in = tf.complex(real_array, imag_array)
  #   real_out = tf.signal.irfft(complex_in)
  #   return real_out


class FftTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(FftModule)
    self.real_array = np.array([
        9., 1., 4.5, -0.3, 10., -1., 5.5, 0.3, 299., 3.5, -0.777, 2, 1.7, 3.5,
        -4.5, 0.0
    ],
                               dtype=np.float32)
    self.imag_array = np.array([
        0., -1., 17.7, 10., 0., -11., 2763, 0., 0., -1.5, 16.8, 100., 0., -111.,
        2.3, 1.
    ],
                               dtype=np.float32)

    # Required since pffft requires a minimum of 32 elements for real ffts.
    self.long_real_array = np.concatenate((self.real_array, self.real_array),
                                          axis=None)

  def test_fft_real(self):

    def fft_real(module):
      module.fft_real(self.real_array, self.imag_array, rtol=1e-4)

    self.compare_backends(fft_real, self._modules)

  def test_fft_imag(self):

    def fft_imag(module):
      module.fft_imag(self.real_array, self.imag_array, rtol=1e-4)

    self.compare_backends(fft_imag, self._modules)

  def test_ifft_real(self):

    def ifft_real(module):
      module.ifft_real(self.real_array, self.imag_array, rtol=1e-4)

    self.compare_backends(ifft_real, self._modules)

  def test_ifft_imag(self):

    def ifft_imag(module):
      module.ifft_imag(self.real_array, self.imag_array, rtol=1e-4)

    self.compare_backends(ifft_imag, self._modules)

  def test_rfft_real(self):

    def rfft_real(module):
      module.rfft_real(self.long_real_array, rtol=1e-4)

    self.compare_backends(rfft_real, self._modules)

  def test_rfft_imag(self):

    def rfft_imag(module):
      module.rfft_imag(self.long_real_array, rtol=1e-4)

    self.compare_backends(rfft_imag, self._modules)


# TODO(natashaknk): Enable IRFFT tests when Linalg on tensors changes land.
#   def test_irfft(self):

#     def irfft(module):
#       module.irfft(self.real_array, self.imag_array, rtol=1e-4)

#     self.compare_backends(irfft, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
