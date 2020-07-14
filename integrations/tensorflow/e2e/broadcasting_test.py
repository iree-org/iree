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
"""Test broadcasting support."""

from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class BroadcastingModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([None], tf.float32),
      tf.TensorSpec([None], tf.float32),
  ])
  def add(self, lhs, rhs):
    return lhs + rhs


@tf_test_utils.compile_module(BroadcastingModule)
class BroadcastingTest(tf_test_utils.SavedModelTestCase):

  def test_add_same_shape(self):
    m = self.get_module()
    dst = m.add(tf.random.uniform([4]), tf.random.uniform([4]))
    dst.print().assert_all_close()


# TODO(silvasean): Make these work.
#   def test_add_broadcast_lhs(self):
#     m = self.get_module()
#     dst = m.add(tf.random.uniform([1]), tf.random.uniform([4]))
#     dst.print().assert_all_close()
#
#   def test_add_broadcast_rhs(self):
#     m = self.get_module()
#     dst = m.add(tf.random.uniform([4]), tf.random.uniform([1]))
#     dst.print().assert_all_close()

if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
