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

from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class Stateful(tf.Module):

  def __init__(self):
    super(Stateful, self).__init__()
    self.counter = tf.Variable(0.0)

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def inc_by(self, x):
    self.counter.assign(self.counter + x)

  @tf.function(input_signature=[])
  def get_state(self):
    return self.counter


@tf_test_utils.compile_module(Stateful)
class StatefulTest(tf_test_utils.SavedModelTestCase):

  def test_stateful(self):
    m = self.get_module()
    m.inc_by(tf.constant(1.))
    m.get_state().print().assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
