# Lint as: python3
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
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

NUM_BATCH = 7
NUM_TIMESTEPS = 24
NUM_UNITS = 10
DYNAMIC_SHAPE = [None, None, NUM_UNITS]
INPUT_SHAPE = [NUM_BATCH, NUM_TIMESTEPS, NUM_UNITS]


class LstmModule(tf.Module):

  def __init__(self):
    super(LstmModule, self).__init__()
    tf_utils.set_random_seed()
    inputs = tf.keras.layers.Input(batch_size=None, shape=DYNAMIC_SHAPE[1:])
    outputs = tf.keras.layers.LSTM(
        units=NUM_UNITS, return_sequences=True)(
            inputs)
    self.m = tf.keras.Model(inputs, outputs)
    self.predict = tf.function(
        input_signature=[tf.TensorSpec(DYNAMIC_SHAPE, tf.float32)])(
            self.m.call)


@tf_test_utils.compile_module(LstmModule, exported_names=["predict"])
class LstmTest(tf_test_utils.TracedModuleTestCase):

  def test_lstm(self):

    def predict(module):
      inputs = tf_utils.ndarange(INPUT_SHAPE)
      module.predict(inputs)

    self.compare_backends(predict)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
