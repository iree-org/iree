# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import unittest

import iree.compiler
import iree.runtime
from iree.runtime.benchmark import (
    benchmark_module,
    BenchmarkTimeoutError,
)


def create_simple_mul_module(instance):
    binary = iree.compiler.compile_str(
        """
      module @test_module {
        func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
          %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
          return %0 : tensor<4xf32>
        }
      }
      """,
        target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
    )
    m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
    return m


def create_large_matmul_module(instance):
    binary = iree.compiler.compile_str(
        """
      module @test_module {
        func.func @large_matmul(%arg0: tensor<4000x4000xf32>, %arg1: tensor<4000x4000xf32>, %arg2: tensor<4000x4000xf32>) -> tensor<4000x4000xf32> {
          %0 = linalg.matmul ins(%arg0, %arg1: tensor<4000x4000xf32>, tensor<4000x4000xf32>)
                    outs(%arg2: tensor<4000x4000xf32>) -> tensor<4000x4000xf32>
          return %0 : tensor<4000x4000xf32>
        }
      }
      """,
        target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
    )
    m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
    return m


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def testBenchmarkModule(self):
        ctx = iree.runtime.SystemContext()
        vm_module = create_simple_mul_module(ctx.instance)
        arg0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arg1 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        benchmark_results = benchmark_module(
            vm_module,
            device=iree.compiler.core.DEFAULT_TESTING_DRIVER,
            inputs=[arg0, arg1],
        )

        self.assertEquals(len(benchmark_results), 1)
        benchmark_time = float(benchmark_results[0].time.split(" ")[0])
        self.assertGreater(benchmark_time, 0)

    def testBenchmarkModuleTimeout(self):
        ctx = iree.runtime.SystemContext()
        vm_module = create_large_matmul_module(ctx.instance)
        arg0 = np.zeros([4000, 4000], dtype=np.float32)
        arg1 = np.zeros([4000, 4000], dtype=np.float32)
        arg2 = np.zeros([4000, 4000], dtype=np.float32)

        with self.assertRaises(BenchmarkTimeoutError):
            _ = benchmark_module(
                vm_module,
                device=iree.compiler.core.DEFAULT_TESTING_DRIVER,
                inputs=[arg0, arg1, arg2],
                timeout=0.1,
            )
