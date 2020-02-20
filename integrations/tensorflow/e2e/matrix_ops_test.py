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
"""Test matrix ops."""

import os
import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf

# TODO(b/147890602)
os.environ["IREE_TEST_BACKENDS"] = "tf"


class MatrixOpsModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([1, 2, 3], tf.float32),
      tf.TensorSpec([3, 4], tf.float32)
  ])
  def batch_matmul(self, x, y):
    return tf.matmul(x, y)


@tf_test_utils.compile_modules(mat=(MatrixOpsModule, ["batch_matmul"]))
class MatrixOpsTest(tf_test_utils.SavedModelTestCase):

  def test_batch_matmul(self):
    a = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                 dtype=np.float32)
    ab = self.modules.mat.all.batch_matmul(a, b)
    output = np.array([[[38, 44, 50, 56], [83, 98, 113, 128]]],
                      dtype=np.float32)
    assert np.allclose(ab, output)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
