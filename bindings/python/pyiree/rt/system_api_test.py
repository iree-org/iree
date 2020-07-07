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

# pylint: disable=unused-variable

import re

from absl.testing import absltest
import numpy as np
from pyiree import compiler
from pyiree import rt


def create_simple_mul_module():
  ctx = compiler.Context()
  input_module = ctx.parse_asm("""
  module @arithmetic {
    func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
          attributes { iree.module.export } {
        %0 = "mhlo.multiply"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
    }
  }
  """)
  binary = input_module.compile()
  m = rt.VmModule.from_flatbuffer(binary)
  return m


class SystemApiTest(absltest.TestCase):

  def test_non_existing_driver(self):
    with self.assertRaisesRegex(RuntimeError,
                                "Could not create any requested driver"):
      config = rt.Config("nothere1,nothere2")

  def test_subsequent_driver(self):
    config = rt.Config("nothere1,vmla")

  def test_empty_dynamic(self):
    ctx = rt.SystemContext()
    self.assertTrue(ctx.is_dynamic)
    self.assertIn("hal", ctx.modules)
    self.assertEqual(ctx.modules.hal.name, "hal")

  def test_empty_static(self):
    ctx = rt.SystemContext(modules=())
    self.assertFalse(ctx.is_dynamic)
    self.assertIn("hal", ctx.modules)
    self.assertEqual(ctx.modules.hal.name, "hal")

  def test_custom_dynamic(self):
    ctx = rt.SystemContext()
    self.assertTrue(ctx.is_dynamic)
    ctx.add_module(create_simple_mul_module())
    self.assertEqual(ctx.modules.arithmetic.name, "arithmetic")
    f = ctx.modules.arithmetic["simple_mul"]
    f_repr = repr(f)
    print(f_repr)
    self.assertRegex(
        f_repr,
        re.escape(
            "(Buffer<float32[4]>, Buffer<float32[4]>) -> (Buffer<float32[4]>)"))

  def test_duplicate_module(self):
    ctx = rt.SystemContext()
    self.assertTrue(ctx.is_dynamic)
    ctx.add_module(create_simple_mul_module())
    with self.assertRaisesRegex(ValueError, "arithmetic"):
      ctx.add_module(create_simple_mul_module())

  def test_static_invoke(self):
    ctx = rt.SystemContext()
    self.assertTrue(ctx.is_dynamic)
    ctx.add_module(create_simple_mul_module())
    self.assertEqual(ctx.modules.arithmetic.name, "arithmetic")
    f = ctx.modules.arithmetic["simple_mul"]
    arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
    arg1 = np.array([4., 5., 6., 7.], dtype=np.float32)
    results = f(arg0, arg1)
    np.testing.assert_allclose(results, [4., 10., 18., 28.])

  def test_load_module(self):
    arithmetic = rt.load_module(create_simple_mul_module())
    arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
    arg1 = np.array([4., 5., 6., 7.], dtype=np.float32)
    results = arithmetic.simple_mul(arg0, arg1)
    np.testing.assert_allclose(results, [4., 10., 18., 28.])


if __name__ == "__main__":
  absltest.main()
