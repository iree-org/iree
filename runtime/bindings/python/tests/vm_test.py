# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pylint: disable=unused-variable

import gc
import logging
import numpy as np
import os
import sys
import tempfile
import traceback
import unittest

import iree.compiler
import iree.runtime

COMPILED_ADD_SCALAR = None


def compile_add_scalar():
    global COMPILED_ADD_SCALAR
    if not COMPILED_ADD_SCALAR:
        COMPILED_ADD_SCALAR = iree.compiler.compile_str(
            """
            func.func @add_scalar(%arg0: i32, %arg1: i32) -> i32 {
              %0 = arith.addi %arg0, %arg1 : i32
              return %0 : i32
            }
            """,
            target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
        )
    return COMPILED_ADD_SCALAR


def create_add_scalar_module(instance):
    binary = compile_add_scalar()
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
        self.device = iree.runtime.get_device(iree.compiler.core.DEFAULT_TESTING_DRIVER)
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
        context = iree.runtime.VmContext(self.instance, modules=[self.hal_module, m])
        logging.info("context: %s", context)

    def test_dynamic_shape_compile(self):
        m = create_simple_dynamic_abs_module(self.instance)
        logging.info("module: %s", m)
        context = iree.runtime.VmContext(self.instance, modules=[self.hal_module, m])
        logging.info("context: %s", context)

    def test_add_scalar_new_abi(self):
        m = create_add_scalar_module(self.instance)
        context = iree.runtime.VmContext(self.instance, modules=[self.hal_module, m])
        f = m.lookup_function("add_scalar")
        finv = iree.runtime.FunctionInvoker(context, self.device, f)
        result = finv(5, 6)
        logging.info("result: %s", result)
        self.assertEqual(result, 11)

    def test_unaligned_buffer_error(self):
        buffer = memoryview(b"foobar")
        with self.assertRaisesRegex(ValueError, "unaligned buffer"):
            # One byte into a heap buffer will never satisfy alignment
            # constraints.
            iree.runtime.VmModule.wrap_buffer(self.instance, buffer[1:])

    def test_from_buffer_unaligned_warns(self):
        binary = compile_add_scalar()
        # One byte into a heap buffer will never satisfy alignment
        # constraints.
        unaligned_binary = memoryview(b"1" + binary)[1:]
        with self.assertWarnsRegex(
            UserWarning, "Making copy of unaligned VmModule buffer"
        ):
            iree.runtime.VmModule.from_buffer(self.instance, unaligned_binary)

    def test_mmap_implicit_unmap(self):
        binary = compile_add_scalar()
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(binary)
            tf.flush()
            vmfb_file_path = tf.name

        # Note that on Windows, an open file cannot be mapped.
        try:
            m = iree.runtime.VmModule.mmap(self.instance, vmfb_file_path)
            context = iree.runtime.VmContext(
                self.instance, modules=[self.hal_module, m]
            )
            f = m.lookup_function("add_scalar")
            finv = iree.runtime.FunctionInvoker(context, self.device, f)
            result = finv(5, 6)
            logging.info("result: %s", result)
            self.assertEqual(result, 11)

            del finv
            del f
            del context
            del m
            gc.collect()
        finally:
            # On Windows, a mapped file cannot be deleted and this will fail if
            # the mapping was not cleaned up properly.
            os.unlink(vmfb_file_path)

    def test_mmap_destroy_callback(self):
        binary = compile_add_scalar()
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(binary)
            tf.flush()
            vmfb_file_path = tf.name

        destroyed = [False]

        def on_destroy():
            print("on_destroy callback")
            try:
                os.unlink(vmfb_file_path)
            except:
                print("exception while unlinking mapped file")
                traceback.print_exc(file=sys.stdout)
                raise
            destroyed[0] = True

        # Note that on Windows, an open file cannot be mapped.
        m = iree.runtime.VmModule.mmap(
            self.instance, vmfb_file_path, destroy_callback=on_destroy
        )
        context = iree.runtime.VmContext(self.instance, modules=[self.hal_module, m])
        f = m.lookup_function("add_scalar")
        finv = iree.runtime.FunctionInvoker(context, self.device, f)
        result = finv(5, 6)
        logging.info("result: %s", result)
        self.assertEqual(result, 11)

        del finv
        del f
        del context
        del m
        gc.collect()
        self.assertTrue(destroyed[0])

    def test_synchronous_dynamic_shape_invoke_function_new_abi(self):
        m = create_simple_dynamic_abs_module(self.instance)
        context = iree.runtime.VmContext(self.instance, modules=[self.hal_module, m])
        f = m.lookup_function("dynamic_abs")
        finv = iree.runtime.FunctionInvoker(context, self.device, f)
        arg0 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
        result = finv(arg0)
        logging.info("result: %s", result)
        np.testing.assert_allclose(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_synchronous_invoke_function_new_abi(self):
        m = create_simple_static_mul_module(self.instance)
        context = iree.runtime.VmContext(self.instance, modules=[self.hal_module, m])
        f = m.lookup_function("simple_mul")
        finv = iree.runtime.FunctionInvoker(context, self.device, f)
        arg0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arg1 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        result = finv(arg0, arg1)
        logging.info("result: %s", result)
        np.testing.assert_allclose(result, [4.0, 10.0, 18.0, 28.0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
