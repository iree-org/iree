#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import tempfile
import unittest

from common.benchmark_suite import BenchmarkCase, BenchmarkSuite


class BenchmarkSuiteTest(unittest.TestCase):

  def test_list_categories(self):
    suite = BenchmarkSuite({
        "suite/TFLite": [],
        "suite/PyTorch": [],
    })

    self.assertEqual(suite.list_categories(), [("PyTorch", "suite/PyTorch"),
                                               ("TFLite", "suite/TFLite")])

  def test_filter_benchmarks_for_category(self):
    case1 = BenchmarkCase(model_name_with_tags="deepnet",
                          bench_mode="1-thread,full-inference",
                          target_arch="CPU-ARMv8",
                          driver="iree-dylib",
                          benchmark_case_dir="case1",
                          benchmark_tool_name="tool")
    case2 = BenchmarkCase(model_name_with_tags="deepnetv2-f32",
                          bench_mode="full-inference",
                          target_arch="GPU-Mali",
                          driver="iree-vulkan",
                          benchmark_case_dir="case2",
                          benchmark_tool_name="tool")
    suite = BenchmarkSuite({
        "suite/TFLite": [case1, case2],
    })

    both_benchmarks = suite.filter_benchmarks_for_category(
        category="TFLite",
        available_drivers=["dylib", "vulkan"],
        cpu_target_arch_filter="cpu-armv8",
        gpu_target_arch_filter="gpu-mali",
        driver_filter=None,
        mode_filter=".*full-inference.*",
        model_name_filter="deepnet.*")
    gpu_benchmarks = suite.filter_benchmarks_for_category(
        category="TFLite",
        available_drivers=["dylib", "vulkan"],
        cpu_target_arch_filter="cpu-unknown",
        gpu_target_arch_filter="gpu-mali",
        driver_filter="vulkan",
        mode_filter=".*full-inference.*",
        model_name_filter="deepnet.*/case2")

    self.assertEqual(both_benchmarks, [case1, case2])
    self.assertEqual(gpu_benchmarks, [case2])

  def test_filter_benchmarks_for_nonexistent_category(self):
    suite = BenchmarkSuite({
        "suite/TFLite": [],
    })

    benchmarks = suite.filter_benchmarks_for_category(
        category="PyTorch",
        available_drivers=[],
        cpu_target_arch_filter="ARMv8",
        gpu_target_arch_filter="Mali-G78")

    self.assertEqual(benchmarks, [])

  def test_load_from_benchmark_suite_dir(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      tflite_dir = os.path.join(tmp_dir, "TFLite")
      pytorch_dir = os.path.join(tmp_dir, "PyTorch")
      BenchmarkSuiteTest.__create_bench(tflite_dir,
                                        model="DeepNet",
                                        bench_mode="4-thread,full",
                                        target_arch="CPU-ARMv8",
                                        driver="iree-dylib",
                                        tool="run-cpu-bench")
      case2 = BenchmarkSuiteTest.__create_bench(pytorch_dir,
                                                model="DeepNetv2",
                                                bench_mode="full-inference",
                                                target_arch="GPU-Mali",
                                                driver="iree-vulkan",
                                                tool="run-gpu-bench")

      suite = BenchmarkSuite.load_from_benchmark_suite_dir(tmp_dir)

      self.assertEqual(suite.list_categories(), [("PyTorch", pytorch_dir),
                                                 ("TFLite", tflite_dir)])
      self.assertEqual(
          suite.filter_benchmarks_for_category(
              category="PyTorch",
              available_drivers=["vulkan"],
              cpu_target_arch_filter="cpu-armv8",
              gpu_target_arch_filter="gpu-mali"), [case2])

  @staticmethod
  def __create_bench(dir_path: str, model: str, bench_mode: str,
                     target_arch: str, driver: str, tool: str):
    case_name = f"{driver}__{target_arch}__{bench_mode}"
    bench_path = os.path.join(dir_path, model, case_name)
    os.makedirs(bench_path)
    with open(os.path.join(bench_path, "tool"), "w") as f:
      f.write(tool)

    return BenchmarkCase(model_name_with_tags=model,
                         bench_mode=bench_mode,
                         target_arch=target_arch,
                         driver=driver,
                         benchmark_case_dir=bench_path,
                         benchmark_tool_name=tool)


if __name__ == "__main__":
  unittest.main()
