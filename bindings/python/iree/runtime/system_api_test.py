# Lint as: python3
# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pylint: disable=unused-variable

import re

from absl import logging
from absl.testing import absltest
import iree.compiler
import iree.runtime
import numpy as np


def create_simple_mul_module():
  binary = iree.compiler.compile_str(
      """
      module @arithmetic {
        func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
              attributes { iree.module.export } {
            %0 = "mhlo.multiply"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
        }
      }
      """,
      input_type="mhlo",
      target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
  )
  m = iree.runtime.VmModule.from_flatbuffer(binary)
  return m


class SystemApiTest(absltest.TestCase):

  def test_non_existing_driver(self):
    with self.assertRaisesRegex(RuntimeError,
                                "Could not create any requested driver"):
      config = iree.runtime.Config("nothere1,nothere2")

  def test_subsequent_driver(self):
    config = iree.runtime.Config("nothere1,dylib")

  def test_empty_dynamic(self):
    ctx = iree.runtime.SystemContext()
    self.assertTrue(ctx.is_dynamic)
    self.assertIn("hal", ctx.modules)
    self.assertEqual(ctx.modules.hal.name, "hal")

  def test_empty_static(self):
    ctx = iree.runtime.SystemContext(vm_modules=())
    self.assertFalse(ctx.is_dynamic)
    self.assertIn("hal", ctx.modules)
    self.assertEqual(ctx.modules.hal.name, "hal")

  def test_custom_dynamic(self):
    ctx = iree.runtime.SystemContext()
    self.assertTrue(ctx.is_dynamic)
    ctx.add_vm_module(create_simple_mul_module())
    self.assertEqual(ctx.modules.arithmetic.name, "arithmetic")
    f = ctx.modules.arithmetic["simple_mul"]
    f_repr = repr(f)
    logging.info("f_repr: %s", f_repr)
    self.assertRegex(
        f_repr,
        re.escape(
            "(Buffer<float32[4]>, Buffer<float32[4]>) -> (Buffer<float32[4]>)"))

  def test_duplicate_module(self):
    ctx = iree.runtime.SystemContext()
    self.assertTrue(ctx.is_dynamic)
    ctx.add_vm_module(create_simple_mul_module())
    with self.assertRaisesRegex(ValueError, "arithmetic"):
      ctx.add_vm_module(create_simple_mul_module())

  def test_static_invoke(self):
    ctx = iree.runtime.SystemContext()
    self.assertTrue(ctx.is_dynamic)
    ctx.add_vm_module(create_simple_mul_module())
    self.assertEqual(ctx.modules.arithmetic.name, "arithmetic")
    f = ctx.modules.arithmetic["simple_mul"]
    arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
    arg1 = np.array([4., 5., 6., 7.], dtype=np.float32)
    results = f(arg0, arg1)
    np.testing.assert_allclose(results, [4., 10., 18., 28.])

  def test_serialize_values(self):
    ctx = iree.runtime.SystemContext()
    self.assertTrue(ctx.is_dynamic)
    ctx.add_vm_module(create_simple_mul_module())
    self.assertEqual(ctx.modules.arithmetic.name, "arithmetic")
    f = ctx.modules.arithmetic["simple_mul"]
    arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
    arg1 = np.array([4., 5., 6., 7.], dtype=np.float32)
    results = f(arg0, arg1)
    inputs, outputs = f.get_serialized_values()
    self.assertEqual(inputs, ("4xf32=1 2 3 4", "4xf32=4 5 6 7"))
    self.assertEqual(outputs, ("4xf32=4 10 18 28",))

  def test_load_vm_module(self):
    arithmetic = iree.runtime.load_vm_module(create_simple_mul_module())
    arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
    arg1 = np.array([4., 5., 6., 7.], dtype=np.float32)
    results = arithmetic.simple_mul(arg0, arg1)
    np.testing.assert_allclose(results, [4., 10., 18., 28.])


if __name__ == "__main__":
  absltest.main()
