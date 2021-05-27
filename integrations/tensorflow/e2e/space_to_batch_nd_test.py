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
"""Space To Batch ND tests."""

from absl import app
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf


class SpaceToBatchModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([1, 8, 2], tf.float32)])
  def batch_to_space_nd(self, x):
    block_shape = [3]
    paddings = [[3, 4]]
    return tf.space_to_batch_nd(x, block_shape, paddings)


class SpaceToBatchTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(SpaceToBatchModule)

  def test_space_to_batch_inference(self):

    def space_to_batch_inference(module):
      x = np.linspace(0, 15, 16, dtype=np.float32)
      x = np.reshape(x, [1, 8, 2])
      module.batch_to_space_nd(x)

    self.compare_backends(space_to_batch_inference, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
