# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
from collections.abc import Callable
import gc
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

    def _create_async_archive_provider(
        self, path: Path, scope: str
    ) -> rt.ParameterProvider:
        index = rt.ParameterIndex()
        index.load(str(path), mode="file_async")
        handle, _ = index.items()[0][1].file_storage
        self.assertTrue(handle.is_async)
        return index.create_provider(scope=scope)

    def _create_archive_provider_after_caller_close(
        self, path: Path, scope: str
    ) -> rt.ParameterProvider:
        index = rt.ParameterIndex()
        with open(path, "rb") as source:
            handle = rt.FileHandle.wrap_fd(source.fileno())
            index.load_from_file_handle(handle, "irpa")
        # The original Python file is closed before any parameter use.
        return index.create_provider(scope=scope)

    def _run_archive_provider_module(
        self,
        directory: Path,
        expected: list[np.ndarray],
        create_provider: Callable[[Path, str], rt.ParameterProvider],
    ) -> None:
        providers = []
        for scope, arrays in (("a", expected[:2]), ("b", expected[2:])):
            path = directory / f"{scope}.irpa"
            rt.save_archive_file(
                {f"{scope}{i}": value for i, value in enumerate(arrays)}, path
            )
            providers.append(create_provider(path, scope))
        parameter_module = rt.create_io_parameters_module(self.instance, *providers)
        del providers
        gc.collect()
        modules = rt.load_vm_modules(
            parameter_module,
            rt.create_hal_module(self.instance, self.device),
            create_mm_test_module(self.instance),
            config=self.config,
        )
        actual = modules[-1].echo()
        for i, (want, got) in enumerate(zip(expected, actual)):
            np.testing.assert_array_equal(want, got, err_msg=f"parameter {i}")

    def _check_archive_provider_module(
        self, create_provider: Callable[[Path, str], rt.ParameterProvider]
    ) -> None:
        expected = [
            np.arange(n, dtype=np.int64) + i * 10 for i, n in enumerate((4, 4, 8, 8))
        ]

        with tempfile.TemporaryDirectory() as td:
            self._run_archive_provider_module(Path(td), expected, create_provider)
            gc.collect()

    def test_async_archive_provider_module(self) -> None:
        self._check_archive_provider_module(self._create_async_archive_provider)

    def test_archive_provider_after_caller_close(self) -> None:
        self._check_archive_provider_module(
            self._create_archive_provider_after_caller_close
        )

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
