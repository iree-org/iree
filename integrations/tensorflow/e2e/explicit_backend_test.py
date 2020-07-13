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
"""Tests explicitly specifying a backend in Python."""

import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class SimpleArithmeticModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    return a * b


@tf_test_utils.compile_module(SimpleArithmeticModule)
class ExplicitBackendTest(tf_test_utils.SavedModelTestCase):

  def test_explicit(self):
    a = np.array([1., 2., 3., 4.], dtype=np.float32)
    b = np.array([400., 5., 6., 7.], dtype=np.float32)

    # Demonstrates simple, one by one invocation of functions against
    # different explicit backends. Individual backends can be accessed off of
    # the module by name ('tf', 'iree_vmla' below).
    tf_c = self.compiled_modules.tf.simple_mul(a, b)
    print("TF Result:", tf_c)
    iree_c = self.compiled_modules.iree_vmla.simple_mul(a, b)
    print("IREE Result:", iree_c)
    self.assertAllClose(tf_c, iree_c)

  def test_multi(self):
    a = np.array([1., 2., 3., 4.], dtype=np.float32)
    b = np.array([400., 5., 6., 7.], dtype=np.float32)

    # Evaluating against multiple backends can be done with the multi() method,
    # which takes a regex string matching backend names. This also returns a
    # MultiResults tuple with actual results keyed by backend name. These also
    # have convenience methods like print() and assert_all_close().
    vmod = self.compiled_modules.multi("tf|iree")
    r = vmod.simple_mul(a, b)
    r.print().assert_all_close()

  def test_get_module(self):
    a = np.array([1., 2., 3., 4.], dtype=np.float32)
    b = np.array([400., 5., 6., 7.], dtype=np.float32)

    # Evaluating against all backends can be done with self.get_module(). This
    # also returns a MultiResults tuple with actual results keyed by backend
    # name.
    r = self.get_module().simple_mul(a, b)
    r.print().assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
