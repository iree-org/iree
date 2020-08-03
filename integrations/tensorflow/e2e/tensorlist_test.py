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

STATIC_SIZE = 20


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

  @tf.function(input_signature=[tf.TensorSpec([STATIC_SIZE], tf.float32)])
  def slice_first_element_with_from_tensor(self, t):
    ta = tf.TensorArray(dtype=tf.float32, size=STATIC_SIZE, element_shape=[])
    ta = ta.unstack(t)
    return ta.read(0)

  @tf.function(
      input_signature=[tf.TensorSpec([STATIC_SIZE, STATIC_SIZE], tf.float32)])
  def slice_first_element_with_from_tensor_high_rank(self, t):
    ta = tf.TensorArray(
        dtype=tf.float32, size=STATIC_SIZE, element_shape=[STATIC_SIZE])
    ta = ta.unstack(t)
    return ta.read(0)

  @tf.function(input_signature=[
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32)
  ])
  def concat_with_tensorlist_stack(self, a, b):
    ta = tf.TensorArray(dtype=tf.float32, size=2, element_shape=[])
    ta = ta.write(0, a)
    ta = ta.write(1, b)
    return ta.stack()


@tf_test_utils.compile_module(TensorListModule)
class TensorListTest(tf_test_utils.TracedModuleTestCase):

  def test_identity_through_tensorlist(self):

    def identity_through_tensorlist(module):
      module.identity_through_tensorlist(np.array(42., dtype=np.float32))

    self.compare_backends(identity_through_tensorlist)

  def test_add_through_tensorlist(self):

    def add_through_tensorlist(module):
      module.add_through_tensorlist(
          np.array(42., dtype=np.float32), np.array(43., dtype=np.float32))

    self.compare_backends(add_through_tensorlist)

  def test_slice_first_element_with_from_tensor(self):

    def slice_first_element_with_from_tensor(module):
      module.slice_first_element_with_from_tensor(
          np.arange(STATIC_SIZE, dtype=np.float32))

    self.compare_backends(slice_first_element_with_from_tensor)

  def test_slice_first_element_with_from_tensor_high_rank(self):

    def slice_first_element_with_from_tensor_high_rank(module):
      module.slice_first_element_with_from_tensor_high_rank(
          tf_utils.ndarange([STATIC_SIZE, STATIC_SIZE]))

    self.compare_backends(slice_first_element_with_from_tensor_high_rank)

  def test_concat_with_tensorlist_stack(self):

    def concat_with_tensorlist_stack(module):
      module.concat_with_tensorlist_stack(
          np.array(42., dtype=np.float32), np.array(43., dtype=np.float32))

    self.compare_backends(concat_with_tensorlist_stack)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
