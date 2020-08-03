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


class ControlFlowModule(tf.Module):

  def __init__(self):
    pass

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def collatz(self, a):
    i = 0.
    while a > 1.:
      i = i + 1.
      if (a % 2.) > 0.:
        a = 3. * a + 1.
      else:
        a = a / 2.
    return i


@tf_test_utils.compile_module(ControlFlowModule)
class ControlFlowTest(tf_test_utils.TracedModuleTestCase):

  def test_short_sequence(self):

    def short_sequence(module):
      input_array = np.array(9., dtype=np.float32)
      module.collatz(input_array)

    self.compare_backends(short_sequence)

  def test_long_sequence(self):

    def long_sequence(module):
      input_array = np.array(178., dtype=np.float32)
      module.collatz(input_array)

    self.compare_backends(long_sequence)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
