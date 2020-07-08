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


class GatherModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4, 8], tf.float32),
      tf.TensorSpec([], tf.int32)
  ])
  def gather_axis0_scalar(self, params, indices):
    return tf.gather(params, indices)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 8], tf.float32),
      tf.TensorSpec([2], tf.int32)
  ])
  def gather_axis0_batch0(self, params, indices):
    return tf.gather(params, indices)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 7, 8], tf.float32),
      tf.TensorSpec([2], tf.int32)
  ])
  def gather_axis1_batch0(self, params, indices):
    return tf.gather(params, indices, axis=1)

  @tf.function(input_signature=[
      tf.TensorSpec([4, 7, 8, 2], tf.float32),
      tf.TensorSpec([4, 1], tf.int32)
  ])
  def gather_axis2_batch1(self, params, indices):
    return tf.gather(params, indices, axis=2, batch_dims=1)


@tf_test_utils.compile_module(GatherModule)
class GatherTest(tf_test_utils.SavedModelTestCase):

  def test_gather_axis0_scalar(self):
    indices = np.array(2, dtype=np.int32)
    params = np.arange(32, dtype=np.float32).reshape(4, 8)
    result = self.get_module().gather_axis0_scalar(params, indices)
    result.print().assert_all_close()

  def test_gather_axis0_batch0(self):
    indices = np.array([2, 3], dtype=np.int32)
    params = np.arange(32, dtype=np.float32).reshape(4, 8)
    result = self.get_module().gather_axis0_batch0(params, indices)
    result.print().assert_all_close()

  def test_gather_axis1_batch0(self):
    indices = np.array([2, 3], dtype=np.int32)
    params = np.arange(4 * 7 * 8, dtype=np.float32).reshape(4, 7, 8)
    result = self.get_module().gather_axis1_batch0(params, indices)
    result.print().assert_all_close()

  def test_gather_axis2_batch1(self):
    indices = np.array([[2], [3], [0], [1]], dtype=np.int32)
    params = np.arange(4 * 7 * 8 * 2, dtype=np.float32).reshape(4, 7, 8, 2)
    result = self.get_module().gather_axis2_batch1(params, indices)
    result.print().assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
