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


class FillModule(tf.Module):

  def __init__(self):
    pass

  @tf.function(input_signature=[
      tf.TensorSpec([2], tf.int32),
      tf.TensorSpec([], tf.float32)
  ])
  def fill(self, dims, value):
    return tf.fill(dims, value)


@tf_test_utils.compile_module(FillModule)
class FillTest(tf_test_utils.SavedModelTestCase):

  def test_fill(self):
    dims = np.array([2, 3], dtype=np.int32)
    value = np.array(9., dtype=np.float32)

    result = self.get_module().fill(dims, value)
    result.assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
