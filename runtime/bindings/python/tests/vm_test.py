# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pylint: disable=unused-variable

import logging
import numpy as np
import unittest

import iree.compiler
import iree.runtime


def create_add_scalar_module(instance):
  binary = iree.compiler.compile_str(
      """
      func.func @add_scalar(%arg0: i32, %arg1: i32) -> i32 {
        %0 = arith.addi %arg0, %arg1 : i32
        return %0 : i32
      }
      """,
      target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
  )
  m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
  return m


def create_simple_static_mul_module(instance):
  binary = iree.compiler.compile_str(
      """
      func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      """,
      target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
  )
  m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
  return m


def create_simple_dynamic_abs_module(instance):
  binary = iree.compiler.compile_str(
      """
      func.func @dynamic_abs(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
        %0 = math.absf %arg0 : tensor<?x?xf32>
        return %0 : tensor<?x?xf32>
      }
      """,
      target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS,
  )
  m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
  return m


class VmTest(unittest.TestCase):

  @classmethod
  def setUp(self):
    self.instance = iree.runtime.VmInstance()
    self.device = iree.runtime.get_device(
        iree.compiler.core.DEFAULT_TESTING_DRIVER)
    self.hal_module = iree.runtime.create_hal_module(self.instance, self.device)

  def test_context_id(self):
    context1 = iree.runtime.VmContext(self.instance)
    context2 = iree.runtime.VmContext(self.instance)
    self.assertNotEqual(context2.context_id, context1.context_id)

  def test_module_basics(self):
    m = create_simple_static_mul_module(self.instance)
    f = m.lookup_function("simple_mul")
    self.assertGreaterEqual(f.ordinal, 0)
    notfound = m.lookup_function("notfound")
    self.assertIs(notfound, None)

  def test_dynamic_module_context(self):
    context = iree.runtime.VmContext(self.instance)
    m = create_simple_static_mul_module(self.instance)
    context.register_modules([self.hal_module, m])

  def test_static_module_context(self):
    m = create_simple_static_mul_module(self.instance)
    logging.info("module: %s", m)
    context = iree.runtime.VmContext(self.instance,
                                     modules=[self.hal_module, m])
    logging.info("context: %s", context)

  def test_dynamic_shape_compile(self):
    m = create_simple_dynamic_abs_module(self.instance)
    logging.info("module: %s", m)
    context = iree.runtime.VmContext(self.instance,
                                     modules=[self.hal_module, m])
    logging.info("context: %s", context)

  def test_add_scalar_new_abi(self):
    m = create_add_scalar_module(self.instance)
    context = iree.runtime.VmContext(self.instance,
                                     modules=[self.hal_module, m])
    f = m.lookup_function("add_scalar")
    finv = iree.runtime.FunctionInvoker(context, self.device, f, tracer=None)
    result = finv(5, 6)
    logging.info("result: %s", result)
    self.assertEqual(result, 11)

  def test_synchronous_dynamic_shape_invoke_function_new_abi(self):
    m = create_simple_dynamic_abs_module(self.instance)
    context = iree.runtime.VmContext(self.instance,
                                     modules=[self.hal_module, m])
    f = m.lookup_function("dynamic_abs")
    finv = iree.runtime.FunctionInvoker(context, self.device, f, tracer=None)
    arg0 = np.array([[-1., 2.], [3., -4.]], dtype=np.float32)
    result = finv(arg0)
    logging.info("result: %s", result)
    np.testing.assert_allclose(result, [[1., 2.], [3., 4.]])

  def test_synchronous_invoke_function_new_abi(self):
    m = create_simple_static_mul_module(self.instance)
    context = iree.runtime.VmContext(self.instance,
                                     modules=[self.hal_module, m])
    f = m.lookup_function("simple_mul")
    finv = iree.runtime.FunctionInvoker(context, self.device, f, tracer=None)
    arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
    arg1 = np.array([4., 5., 6., 7.], dtype=np.float32)
    result = finv(arg0, arg1)
    logging.info("result: %s", result)
    np.testing.assert_allclose(result, [4., 10., 18., 28.])


if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  unittest.main()
