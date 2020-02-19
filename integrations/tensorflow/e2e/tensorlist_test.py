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

from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class TensorListModule(tf.Module):

  def __init__(self):
    pass

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def identity_through_tensorlist(self, x):
    ta = tf.TensorArray(dtype=tf.float32, size=1, element_shape=[])
    ta = ta.write(0, x)
    return ta.read(0)

  @tf.function(input_signature=[
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32)
  ])
  def add_through_tensorlist(self, a, b):
    ta = tf.TensorArray(dtype=tf.float32, size=2, element_shape=[])
    ta = ta.write(0, a)
    ta = ta.write(1, b)
    return ta.read(0) + ta.read(1)


# TODO(b/146900329): Triage op coverage for vulkan backend.
@tf_test_utils.compile_modules(
    backends=["tf", "iree_interpreter"], tensorlist=TensorListModule)
class TensorListTest(tf_test_utils.SavedModelTestCase):

  def test_identity_through_tensorlist(self):
    m = self.modules.tensorlist.all
    result = m.identity_through_tensorlist(tf.constant(42.))
    result.print().assert_all_close()

  def test_add_through_tensorlist(self):
    m = self.modules.tensorlist.all
    result = m.add_through_tensorlist(tf.constant(42.), tf.constant(43.))
    result.print().assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
