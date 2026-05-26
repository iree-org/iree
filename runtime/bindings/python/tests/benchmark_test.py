# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import unittest
import tempfile
from pathlib import Path

import iree.compiler
import iree.runtime
from iree.runtime.benchmark import (
    _build_benchmark_args as build_benchmark_args,
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


def create_multiple_entry_functions_module(instance):
    binary = iree.compiler.compile_str(
        """
      module @test_module {
        func.func @entry_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
          %0 = math.absf %arg0 : tensor<4xf32>
          return %0 : tensor<4xf32>
        }
        func.func @entry_2(%arg0: tensor<2xf32>) -> tensor<2xf32> {
          %0 = math.absf %arg0 : tensor<2xf32>
          return %arg0 : tensor<2xf32>
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

    def testBuildBenchmarkArgs(self):
        args, flatbuffer = build_benchmark_args(
            module="test_module.vmfb",
            entry_function="test_func",
            inputs=[np.array([1.0, 2.0], dtype=np.float32)],
            # kwargs should be passed through as --{k}={v}
            device="test_device",
            extra_arg="extra_value",
        )
        ref_args = [
            iree.runtime.benchmark_exe(),
            "--module=test_module.vmfb",
            "--function=test_func",
            "--device=test_device",
            "--extra_arg=extra_value",
            "--input=2xf32=1.0,2.0",
        ]
        self.assertEqual(args, ref_args)
        self.assertIsNone(flatbuffer)

    def testBuildBenchmarkArgsInputs(self):
        all_inputs = [
            # empty
            [],
            # single input
            [np.ones([2], dtype=np.float16)],
            # multiple inputs
            [np.ones([2], dtype=np.float64), np.ones([3], dtype=np.int8)],
            # string input
            ["input_string"],
            # explicit values
            [np.array([0, 1, 2, 3], dtype=np.uint8)],
        ]
        ref_args = [
            [],
            ["--input=2xf16=1.0"],
            ["--input=2xf64=1.0", "--input=3xi8=1"],
            ["--input=input_string"],
            ["--input=4xi8=0,1,2,3"],
        ]
        for inp, ref in zip(all_inputs, ref_args):
            args, flatbuffer = build_benchmark_args(
                module="test_module.vmfb",
                entry_function="test_func",
                inputs=inp,
            )
            self.assertEqual(args[3:], ref)
            self.assertIsNone(flatbuffer)

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

        self.assertEqual(len(benchmark_results), 1)
        benchmark_time = float(benchmark_results[0].time.split(" ")[0])
        self.assertGreater(benchmark_time, 0)

    def testBenchmarkModuleFromFilePath(self):
        ctx = iree.runtime.SystemContext()
        vm_module = create_simple_mul_module(ctx.instance)
        with tempfile.TemporaryDirectory() as tmp_dir:
            module_file_path = Path(tmp_dir) / "module.vmfb"
            with open(module_file_path, "wb") as f:
                f.write(vm_module.stashed_flatbuffer_blob)

            arg0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            arg1 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

            benchmark_results = benchmark_module(
                module_file_path,
                entry_function="simple_mul",
                device=iree.compiler.core.DEFAULT_TESTING_DRIVER,
                inputs=[arg0, arg1],
            )

        self.assertEqual(len(benchmark_results), 1)
        benchmark_time = float(benchmark_results[0].time.split(" ")[0])
        self.assertGreater(benchmark_time, 0)

    def testBenchmarkModuleWithEntryFunction(self):
        ctx = iree.runtime.SystemContext()
        vm_module = create_multiple_entry_functions_module(ctx.instance)
        arg1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arg2 = np.array([1.0, 2.0], dtype=np.float32)

        benchmark_results_1 = benchmark_module(
            vm_module,
            entry_function="entry_1",
            device=iree.compiler.core.DEFAULT_TESTING_DRIVER,
            inputs=[arg1],
        )
        self.assertEqual(len(benchmark_results_1), 1)

        benchmark_results_2 = benchmark_module(
            vm_module,
            entry_function="entry_2",
            device=iree.compiler.core.DEFAULT_TESTING_DRIVER,
            inputs=[arg2],
        )
        self.assertEqual(len(benchmark_results_2), 1)

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


if __name__ == "__main__":
    unittest.main()
