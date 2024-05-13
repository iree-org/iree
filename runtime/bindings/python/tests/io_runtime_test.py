# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import logging
import numpy as np
from pathlib import Path
import tempfile
import unittest

import iree.compiler
import iree.runtime as rt


TEST_COMPILED = None
TEST_ASM = r"""
util.global private @a0 = #flow.parameter.named<"a"::"a0"> : tensor<4xi64>
util.global private @a1 = #flow.parameter.named<"a"::"a1"> : tensor<4xi64>
util.global private @b0 = #flow.parameter.named<"b"::"b0"> : tensor<8xi64>
util.global private @b1 = #flow.parameter.named<"b"::"b1"> : tensor<8xi64>
func.func @echo() -> (tensor<4xi64>, tensor<4xi64>, tensor<8xi64>, tensor<8xi64>) {
  %a0 = util.global.load @a0 : tensor<4xi64>
  %a1 = util.global.load @a1 : tensor<4xi64>
  %b0 = util.global.load @b0 : tensor<8xi64>
  %b1 = util.global.load @b1 : tensor<8xi64>
  return %a0, %a1, %b0, %b1 : tensor<4xi64>, tensor<4xi64>, tensor<8xi64>, tensor<8xi64>
}
"""


def compile_mm_test():
    global TEST_COMPILED
    if not TEST_COMPILED:
        TEST_COMPILED = iree.compiler.compile_str(
            TEST_ASM,
            target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
        )
    return TEST_COMPILED


def create_mm_test_module(instance):
    binary = compile_mm_test()
    return rt.VmModule.copy_buffer(instance, binary)


def create_index_from_arrays(**kwargs) -> rt.ParameterIndex:
    idx = rt.ParameterIndex()
    for key, value in kwargs.items():
        idx.add_buffer(key, value)
    return idx


class ParameterTest(unittest.TestCase):
    def setUp(self):
        self.instance = rt.VmInstance()
        self.device = rt.get_device(iree.compiler.core.DEFAULT_TESTING_DRIVER)
        self.config = rt.Config(device=self.device)

    def test_index_provider_module(self):
        a0 = np.asarray([1] * 4, dtype=np.int64)
        a1 = np.asarray([2] * 4, dtype=np.int64)
        b0 = np.asarray([3] * 8, dtype=np.int64)
        b1 = np.asarray([4] * 8, dtype=np.int64)
        idx_a = create_index_from_arrays(a0=a0, a1=a1)
        idx_b = create_index_from_arrays(b0=b0, b1=b1)
        modules = rt.load_vm_modules(
            rt.create_io_parameters_module(
                self.instance,
                idx_a.create_provider(scope="a"),
                idx_b.create_provider(scope="b"),
            ),
            rt.create_hal_module(self.instance, self.device),
            create_mm_test_module(self.instance),
            config=self.config,
        )
        m = modules[-1]
        a0_actual, a1_actual, b0_actual, b1_actual = m.echo()
        np.testing.assert_array_equal(a0, a0_actual)
        np.testing.assert_array_equal(a1, a1_actual)
        np.testing.assert_array_equal(b0, b0_actual)
        np.testing.assert_array_equal(b1, b1_actual)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
