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
"""Test scatter update behavior for tensorflow."""


class ScatterUpdateModule(tf.Module):

  def __init__(self):
    pass

  @tf.function(input_signature=[
      tf.TensorSpec([8], tf.int32),
      tf.TensorSpec([3, 1], tf.int32),
      tf.TensorSpec([3], tf.int32)
  ])
  def scatter_update_1D(self, tensor, indices, updates):
    return tf.tensor_scatter_nd_update(tensor, indices, updates)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 3], tf.int32),
      tf.TensorSpec([3, 2], tf.int32),
      tf.TensorSpec([3], tf.int32)
  ])
  def scatter_update_2D(self, tensor, indices, updates):
    return tf.tensor_scatter_nd_update(tensor, indices, updates)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 3], tf.int32),
      tf.TensorSpec([1, 1], tf.int32),
      tf.TensorSpec([1, 3], tf.int32)
  ])
  def scatter_update_2D_slice(self, tensor, indices, updates):
    return tf.tensor_scatter_nd_update(tensor, indices, updates)


@tf_test_utils.compile_module(ScatterUpdateModule)
class ScatterUpdateTest(tf_test_utils.TracedModuleTestCase):

  def test_scatter_update_1D(self):

    def scatter_update_1D(module):
      tensor = np.ones([8], dtype=np.int32)
      indices = np.array([[4], [5], [6]], dtype=np.int32)
      updates = np.array([9, 10, 11], dtype=np.int32)
      module.scatter_update_1D(tensor, indices, updates)

    self.compare_backends(scatter_update_1D)

  def test_scatter_update_2D(self):

    def scatter_update_2D(module):
      tensor = np.ones([4, 3], dtype=np.int32)
      indices = np.array([[1, 0], [2, 1], [3, 2]], dtype=np.int32)
      updates = np.array([2, 5, 8], dtype=np.int32)
      module.scatter_update_2D(tensor, indices, updates)

    self.compare_backends(scatter_update_2D)

  def test_scatter_update_2D_slice(self):

    def scatter_update_2D_slice(module):
      tensor = np.ones([4, 3], dtype=np.int32)
      indices = np.array([[1]], dtype=np.int32)
      updates = np.array([[2, 3, 4]], dtype=np.int32)
      module.scatter_update_2D_slice(tensor, indices, updates)

    self.compare_backends(scatter_update_2D_slice)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
