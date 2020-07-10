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

import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

HIDDEN_1_DIM = 256
HIDDEN_2_DIM = 256
INPUT_DIM = 728  # 28 * 28
CLASSES = 10


class Mlp(tf.Module):

  def __init__(self,
               hidden_1_dim=256,
               hidden_2_dim=256,
               input_dim=28 * 28,
               classes=10):
    super().__init__()
    tf_utils.set_random_seed()
    self.hidden_1_dim = hidden_1_dim
    self.hidden_2_dim = hidden_2_dim
    self.input_dim = input_dim
    self.classes = classes
    self.h1_weights = tf.Variable(tf.random.normal([input_dim, hidden_1_dim]))
    self.h2_weights = tf.Variable(
        tf.random.normal([hidden_1_dim, hidden_2_dim]))
    self.out_weights = tf.Variable(tf.random.normal([hidden_2_dim, classes]))
    self.h1_bias = tf.Variable(tf.random.normal([hidden_1_dim]))
    self.h2_bias = tf.Variable(tf.random.normal([hidden_2_dim]))
    self.out_bias = tf.Variable(tf.random.normal([classes]))

    # Compile with dynamic batch dim.
    self.predict = tf.function(
        input_signature=[tf.TensorSpec([None, self.input_dim])])(
            self.predict)

  def mlp(self, x):
    layer_1 = tf.sigmoid(tf.add(tf.matmul(x, self.h1_weights), self.h1_bias))
    layer_2 = tf.sigmoid(
        tf.add(tf.matmul(layer_1, self.h2_weights), self.h2_bias))
    return tf.sigmoid(
        tf.add(tf.matmul(layer_2, self.out_weights), self.out_bias))

  def predict(self, x):
    return tf.nn.softmax(self.mlp(x))


@tf_test_utils.compile_module(Mlp, exported_names=["predict"])
class DynamicMlpTest(tf_test_utils.SavedModelTestCase):

  def test_dynamic_batch(self):
    m = self.get_module()
    np.random.seed(12345)
    x = np.random.random([3, 28 * 28]).astype(np.float32) * 1e-3
    m.predict(x).print().assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
