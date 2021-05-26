# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from absl import app
from iree.tf.support import tf_test_utils
import numpy as np
import tensorflow.compat.v2 as tf


class FftModule(tf.Module):

  complex_input_signature = [
      tf.TensorSpec([16], tf.float32),
      tf.TensorSpec([16], tf.float32)
  ]

  @tf.function(input_signature=complex_input_signature)
  def fft(self, real_array, imag_array):
    complex_in = tf.complex(real_array, imag_array)
    complex_out = tf.signal.fft(complex_in)
    return tf.math.real(complex_out), tf.math.imag(complex_out)

  @tf.function(input_signature=complex_input_signature)
  def ifft(self, real_array, imag_array):
    complex_in = tf.complex(real_array, imag_array)
    complex_out = tf.signal.ifft(complex_in)
    return tf.math.real(complex_out), tf.math.imag(complex_out)

  @tf.function(input_signature=[tf.TensorSpec([32], tf.float32)])
  def rfft(self, real_array):
    complex_out = tf.signal.rfft(real_array)
    return tf.math.real(complex_out), tf.math.imag(complex_out)

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

  # yapf: disable
  def test_fft(self):
    def fft(module):
      module.fft(self.real_array, self.imag_array, rtol=1e-4)
    self.compare_backends(fft, self._modules)

  def test_ifft(self):
    def ifft(module):
      module.ifft(self.real_array, self.imag_array, rtol=1e-4)
    self.compare_backends(ifft, self._modules)

  def test_rfft(self):
    def rfft(module):
      module.rfft(self.long_real_array, rtol=1e-4)
    self.compare_backends(rfft, self._modules)

# TODO(natashaknk): Enable IRFFT tests when Linalg on tensors changes land.
#   def test_irfft(self):
#     def irfft(module):
#       module.irfft(self.real_array, self.imag_array, rtol=1e-4)
#     self.compare_backends(irfft, self._modules)
# yapf: enable


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()

if __name__ == '__main__':
  app.run(main)
