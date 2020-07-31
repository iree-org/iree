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

import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class ComplexModule(tf.Module):

  def __init__(self):
    pass

  @tf.function(input_signature=[
      tf.TensorSpec([2], tf.float32),
      tf.TensorSpec([2], tf.float32)
  ])
  def complex_exp(self, real, imag):
    tensor = tf.complex(real, imag)
    exp = tf.exp(tensor)
    return tf.math.real(exp)


@tf_test_utils.compile_module(ComplexModule)
class ComplexTest(tf_test_utils.TracedModuleTestCase):

  def test_complex(self):

    def complex_exp(module):
      real = np.array([2., 3.], dtype=np.float32)
      imag = np.array([-1., 0.4], dtype=np.float32)
      module.complex_exp(real, imag)

    self.compare_backends(complex_exp)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
