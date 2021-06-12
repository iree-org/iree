# Lint as: python3
# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pylint: disable=unused-variable

from absl import logging
from absl.testing import absltest
import iree.compiler
import iree.runtime
import numpy as np


def create_add_scalar_module():
  binary = iree.compiler.compile_str(
      """
      func @add_scalar(%arg0: i32, %arg1: i32) -> i32 attributes { iree.module.export } {
        %0 = addi %arg0, %arg1 : i32
        return %0 : i32
      }
      """,
      input_type="mhlo",
      target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
  )
  m = iree.runtime.VmModule.from_flatbuffer(binary)
  return m


def create_simple_static_mul_module():
  binary = iree.compiler.compile_str(
      """
      func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
            attributes { iree.module.export } {
          %0 = "mhlo.multiply"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          return %0 : tensor<4xf32>
      }
      """,
      input_type="mhlo",
      target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
  )
  m = iree.runtime.VmModule.from_flatbuffer(binary)
  return m


def create_simple_dynamic_abs_module():
  # TODO(laurenzo): Compile for more backends as dynamic shapes come online.
  target_backends = iree.compiler.DEFAULT_TESTING_BACKENDS
  binary = iree.compiler.compile_str(
      """
      func @simple_mul(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32>
            attributes { iree.module.export } {
          %0 = "mhlo.abs"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
          return %0 : tensor<?x?xf32>
      }
      """,
      input_type="mhlo",
      target_backends=target_backends,
  )
  m = iree.runtime.VmModule.from_flatbuffer(binary)
  return m


class VmTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    driver_names = iree.runtime.HalDriver.query()
    logging.info("driver_names: %s", driver_names)
    cls.driver = iree.runtime.HalDriver.create(
        iree.compiler.core.DEFAULT_TESTING_DRIVER)
    cls.device = cls.driver.create_default_device()
    cls.hal_module = iree.runtime.create_hal_module(cls.device)

  def test_variant_list(self):
    l = iree.runtime.VmVariantList(5)
    logging.info("variant_list: %s", l)
    self.assertEqual(l.size, 0)

  def test_variant_list_buffers(self):
    ET = iree.runtime.HalElementType
    for dt, et in ((np.int8, ET.SINT_8), (np.int16, ET.SINT_16),
                   (np.int32, ET.SINT_32), (np.int64, ET.SINT_64),
                   (np.uint8, ET.UINT_8), (np.uint16, ET.UINT_16),
                   (np.uint32, ET.UINT_32), (np.uint64, ET.UINT_64),
                   (np.float32, ET.FLOAT_32), (np.float64, ET.FLOAT_64)):
      # TODO: Unimplemented: (np.float16, ET.FLOAT_16)
      lst = iree.runtime.VmVariantList(5)
      ary1 = np.asarray([1, 2, 3, 4], dtype=dt)
      lst.push_buffer_view(self.device, ary1, et)
      ary2 = lst.get_as_ndarray(0)
      np.testing.assert_array_equal(ary1, ary2)
      with self.assertRaises(IndexError):
        lst.get_as_ndarray(1)

  def test_variant_list_list(self):
    lst1 = iree.runtime.VmVariantList(5)
    lst2 = iree.runtime.VmVariantList(5)
    lst1.push_list(lst2)
    self.assertEqual("<VmVariantList(1): [List[]]>", str(lst1))
    lstout = lst1.get_as_list(0)
    self.assertEqual("<VmVariantList(0): []>", str(lstout))
    with self.assertRaises(IndexError):
      lst1.get_as_list(1)

  def test_context_id(self):
    instance = iree.runtime.VmInstance()
    context1 = iree.runtime.VmContext(instance)
    context2 = iree.runtime.VmContext(instance)
    self.assertGreater(context2.context_id, context1.context_id)

  def test_module_basics(self):
    m = create_simple_static_mul_module()
    f = m.lookup_function("simple_mul")
    self.assertGreaterEqual(f.ordinal, 0)
    notfound = m.lookup_function("notfound")
    self.assertIs(notfound, None)

  def test_dynamic_module_context(self):
    instance = iree.runtime.VmInstance()
    context = iree.runtime.VmContext(instance)
    m = create_simple_static_mul_module()
    context.register_modules([self.hal_module, m])

  def test_static_module_context(self):
    m = create_simple_static_mul_module()
    logging.info("module: %s", m)
    instance = iree.runtime.VmInstance()
    logging.info("instance: %s", instance)
    context = iree.runtime.VmContext(instance, modules=[self.hal_module, m])
    logging.info("context: %s", context)

  def test_dynamic_shape_compile(self):
    m = create_simple_dynamic_abs_module()
    logging.info("module: %s", m)
    instance = iree.runtime.VmInstance()
    logging.info("instance: %s", instance)
    context = iree.runtime.VmContext(instance, modules=[self.hal_module, m])
    logging.info("context: %s", context)

  def test_add_scalar_new_abi(self):
    # TODO: Enable with new ABI.
    return
    m = create_add_scalar_module()
    instance = iree.runtime.VmInstance()
    context = iree.runtime.VmContext(instance, modules=[self.hal_module, m])
    f = m.lookup_function("add_scalar")
    finv = iree.runtime.FunctionInvoker(context, self.device, f)
    result = finv(5, 6)
    logging.info("result: %s", result)
    self.assertEqual(result, 11)

  def test_synchronous_dynamic_shape_invoke_function_new_abi(self):
    # TODO: Enable with new ABI.
    return
    m = create_simple_dynamic_abs_module()
    instance = iree.runtime.VmInstance()
    context = iree.runtime.VmContext(instance, modules=[self.hal_module, m])
    f = m.lookup_function("simple_mul")
    finv = iree.runtime.FunctionInvoker(context, self.device, f)
    arg0 = np.array([[-1., 2.], [3., -4.]], dtype=np.float32)
    result = finv(arg0)
    logging.info("result: %s", result)
    np.testing.assert_allclose(result, [[1., 2.], [3., 4.]])

  def test_synchronous_invoke_function_new_abi(self):
    # TODO: Enable with new ABI.
    return
    m = create_simple_static_mul_module()
    instance = iree.runtime.VmInstance()
    context = iree.runtime.VmContext(instance, modules=[self.hal_module, m])
    f = m.lookup_function("simple_mul")
    finv = iree.runtime.FunctionInvoker(context, self.device, f)
    arg0 = np.array([1., 2., 3., 4.], dtype=np.float32)
    arg1 = np.array([4., 5., 6., 7.], dtype=np.float32)
    result = finv(arg0, arg1)
    logging.info("result: %s", result)
    np.testing.assert_allclose(result, [4., 10., 18., 28.])


if __name__ == "__main__":
  absltest.main()
