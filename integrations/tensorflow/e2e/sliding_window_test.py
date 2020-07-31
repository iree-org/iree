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

TIME_SIZE = 3
FEATURE_SIZE = 2
BATCH_SIZE = 1


class SlidingWindow(tf.keras.layers.Layer):
  # It is another version of a ring buffer
  # during call() it appends new update and remove the oldest one

  def __init__(self, state_shape=None, **kwargs):
    super(SlidingWindow, self).__init__(**kwargs)

    self.state_shape = state_shape

  def build(self, input_shape):
    super(SlidingWindow, self).build(input_shape)

    self.states = self.add_weight(
        name="states",
        shape=self.state_shape,  # [batch, time, feature]
        trainable=False,
        initializer=tf.zeros_initializer)

  def call(self, inputs):

    # [batch_size, 1, feature_dim]
    inputs_time = tf.keras.backend.expand_dims(inputs, -2)

    # remove latest row [batch_size, (memory_size-1), feature_dim]
    memory = self.states[:, 1:self.state_shape[1], :]

    # add new row [batch_size, memory_size, feature_dim]
    memory = tf.keras.backend.concatenate([memory, inputs_time], 1)

    self.states.assign(memory)

    return self.states

  def get_config(self):
    config = {
        "state_shape": self.state_shape,
    }
    base_config = super(SlidingWindow, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SlidingWindowModule(tf.Module):

  def __init__(self):
    super(SlidingWindowModule, self).__init__()
    state_shape = [BATCH_SIZE, TIME_SIZE, FEATURE_SIZE]
    self.sw = SlidingWindow(state_shape=state_shape)

  @tf.function(
      input_signature=[tf.TensorSpec([BATCH_SIZE, FEATURE_SIZE], tf.float32)])
  def predict(self, x):
    return self.sw(x)


@tf_test_utils.compile_module(SlidingWindowModule, exported_names=["predict"])
class SlidingWindowTest(tf_test_utils.TracedModuleTestCase):

  def test_sliding_window(self):

    def sliding_window(module):
      input1 = np.array([[1.0, 2.0]], dtype=np.float32)
      result1 = module.predict(input1)
      # output1 = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 2.0]], dtype=np.float32)

      input2 = np.array([[3.0, 4.0]], dtype=np.float32)
      result2 = module.predict(input2)
      # output2 = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    self.compare_backends(sliding_window)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
