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
"""Several baseline e2e simple arithmetic tests."""

import numpy as np
from pyiree import tf_test_utils
import tensorflow.compat.v2 as tf


class SimpleArithmeticModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    return a * b

  @tf.function(input_signature=[
      tf.TensorSpec([128, 3072], tf.float32),
      tf.TensorSpec([3072, 256], tf.float32),
  ])
  def simple_matmul(self, a, b):
    return tf.matmul(a, b)


@tf_test_utils.compile_modules(simple_arithmetic=SimpleArithmeticModule)
class SimpleArithmeticTest(tf_test_utils.SavedModelTestCase):

  def test_simple_mul_explicit(self):
    # Demonstrates simple, one by one invocation of functions against
    # different explicit backends.
    a = np.array([1., 2., 3., 4.], dtype=np.float32)
    b = np.array([400., 5., 6., 7.], dtype=np.float32)
    # Individual backends can be accessed off of the module by name ('tf,
    # 'iree_interpreter' below).
    tf_c = self.modules.simple_arithmetic.tf.simple_mul(a, b)
    print("TF Result:", tf_c)
    iree_c = self.modules.simple_arithmetic.iree_interpreter.simple_mul(a, b)
    print("IREE Result:", iree_c)
    self.assertAllClose(tf_c, iree_c)

  def test_simple_mul_multi(self):
    a = np.array([1., 2., 3., 4.], dtype=np.float32)
    b = np.array([400., 5., 6., 7.], dtype=np.float32)

    # Evaluating against multiple backends can be done with the multi() method,
    # which takes a regex string matching backend names. This also returns a
    # MultiResults tuple with actual results keyed by backend name. These also
    # have convenience methods like print() and assert_all_close().
    vmod = self.modules.simple_arithmetic.multi("tf|iree")
    r = vmod.simple_mul(a, b)
    r.print().assert_all_close()

  def test_matmul(self):
    np.random.seed(12345)
    # Note: scaling by a small value to increase numerical stability.
    a = np.random.random((128, 3072)).astype(np.float32) * 1e-3
    b = np.random.random((3072, 256)).astype(np.float32) * 1e-3
    # Evaluating against all backends can be done with the special 'all'
    # backend name. This also returns a MultiResults tuple with actual results
    # keyed by backend name.
    r = self.modules.simple_arithmetic.all.simple_matmul(a, b)
    r.print().assert_all_close()

  def test_disagreement(self):
    a = np.array([1., 2., 3., 4.], dtype=np.float32)
    b = np.array([400., 5., 6., 7.], dtype=np.float32)

    vmod = self.modules.simple_arithmetic.all
    r = vmod.simple_mul(a, b)
    # Using a negative atol will cause disagreement even if identical.
    with self.assertRaisesRegex(AssertionError, "Disagreements"):
      r.print().assert_all_close(atol=-3.0)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
