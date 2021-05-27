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


class RangeModule(tf.Module):

  def __init__(self):
    pass

  @tf.function(input_signature=[
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32)
  ])
  def range(self, start, stop, delta):
    return tf.range(start, stop, delta)


class RangeTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(RangeModule)

  def test_range(self):

    def range(module):
      start = np.array(3., dtype=np.float32)
      stop = np.array(12., dtype=np.float32)
      delta = np.array(3, dtype=np.float32)
      result = module.range(start, stop, delta)

    self.compare_backends(range, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
