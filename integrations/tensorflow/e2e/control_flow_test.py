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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
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
class ControlFlowTest(tf_test_utils.SavedModelTestCase):

  def test_short_sequence(self):
    input_array = numpy.array(9., dtype=numpy.float32)
    result = self.get_module().collatz(input_array)
    result.print().assert_all_close()

  def test_long_sequence(self):
    input_array = numpy.array(178., dtype=numpy.float32)
    result = self.get_module().collatz(input_array)
    result.print().assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
