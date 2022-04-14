# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import tempfile
import unittest

from typing import Optional
from common.benchmark_suite import BenchmarkCase, BenchmarkSuite
from common.benchmark_driver import BenchmarkDriver
from common.benchmark_config import BENCHMARK_RESULTS_REL_PATH, CAPTURES_REL_PATH, BenchmarkConfig, TraceCaptureConfig
from common.benchmark_definition import DeviceInfo, PlatformType


class FakeBenchmarkDriver(BenchmarkDriver):

  def __init__(self,
               *args,
               raise_exception_on_case: Optional[str] = None,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.raise_exception_on_case = raise_exception_on_case

  def run_benchmark_case(self, benchmark_case: BenchmarkCase,
                         benchmark_results_filename: Optional[str],
                         capture_filename: Optional[str]) -> None:
    if (self.raise_exception_on_case is not None and
        self.raise_exception_on_case in benchmark_case.benchmark_case_dir):
      raise Exception("fake exception")

    if benchmark_results_filename:
      with open(benchmark_results_filename, "w") as f:
        f.write(json.dumps({
            "context": "fake_context",
            "benchmarks": [],
        }))
    if capture_filename:
      with open(capture_filename, "w") as f:
        f.write("{}")


class BenchmarkDriverTest(unittest.TestCase):

  def setUp(self):
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.root_dir = tempfile.TemporaryDirectory()

    with open(os.path.join(self.tmp_dir.name, "build_config.txt"), "w") as f:
      f.write("IREE_HAL_DRIVER_DYLIB=ON\n")
      f.write("IREE_HAL_DRIVER_DYLIB_SYNC=ON\n")

    self.config = BenchmarkConfig(
        root_benchmark_dir=self.root_dir.name,
        benchmark_results_dir=os.path.join(self.tmp_dir.name,
                                           BENCHMARK_RESULTS_REL_PATH),
        git_commit_hash="abcd",
        normal_benchmark_tool_dir=self.tmp_dir.name,
        trace_capture_config=TraceCaptureConfig(
            traced_benchmark_tool_dir=self.tmp_dir.name,
            trace_capture_tool=os.path.join(self.tmp_dir.name, "capture_tool"),
            capture_tarball="captures.tar",
            capture_tmp_dir=os.path.join(self.tmp_dir.name, CAPTURES_REL_PATH)))

    self.device_info = DeviceInfo(PlatformType.LINUX, "Unknown", "arm64-v8a",
                                  ["sha2"], "Mali-G78")

    case1 = BenchmarkCase(model_name_with_tags="DeepNet",
                          bench_mode="1-thread,full-inference",
                          target_arch="CPU-ARM64-v8A",
                          driver="iree-dylib",
                          benchmark_case_dir="case1",
                          benchmark_tool_name="tool")
    case2 = BenchmarkCase(model_name_with_tags="DeepNetv2-f32",
                          bench_mode="full-inference",
                          target_arch="CPU-ARM64-v8A",
                          driver="iree-dylib-sync",
                          benchmark_case_dir="case2",
                          benchmark_tool_name="tool")
    self.benchmark_suite = BenchmarkSuite({
        "suite/TFLite": [case1, case2],
    })

  def tearDown(self) -> None:
    self.root_dir.cleanup()
    self.tmp_dir.cleanup()

  def test_add_previous_benchmarks_and_captures(self):
    driver = BenchmarkDriver(self.device_info, self.config,
                             self.benchmark_suite)
    os.makedirs(os.path.join(self.tmp_dir.name, BENCHMARK_RESULTS_REL_PATH))
    os.makedirs(os.path.join(self.tmp_dir.name, CAPTURES_REL_PATH))
    benchmark_filename = os.path.join(
        self.tmp_dir.name, BENCHMARK_RESULTS_REL_PATH,
        "MobileNetv2 [fp32,imagenet] (TFLite) big-core,full-inference with IREE-Dylib @ Pixel-4 (CPU-ARMv8.2-A).json"
    )
    capture_filename = os.path.join(
        self.tmp_dir.name, CAPTURES_REL_PATH,
        "MobileNetv2 [fp32,imagenet] (TFLite) big-core,full-inference with IREE-Dylib @ Pixel-4 (CPU-ARMv8.2-A).tracy"
    )
    with open(os.path.join(benchmark_filename), "w") as f:
      f.write("")
    with open(os.path.join(capture_filename), "w") as f:
      f.write("")

    driver.add_previous_benchmarks_and_captures(self.tmp_dir.name)

    self.assertEqual(driver.get_benchmark_result_filenames(),
                     [benchmark_filename])
    self.assertEqual(driver.get_capture_filenames(), [capture_filename])

  def test_run(self):
    driver = FakeBenchmarkDriver(self.device_info, self.config,
                                 self.benchmark_suite)

    driver.run()

    self.assertEqual(driver.get_benchmark_results().commit, "abcd")
    self.assertEqual(len(driver.get_benchmark_results().benchmarks), 2)
    self.assertEqual(driver.get_benchmark_results().benchmarks[0].context,
                     "fake_context")
    self.assertEqual(driver.get_benchmark_result_filenames(), [
        os.path.join(
            self.tmp_dir.name, BENCHMARK_RESULTS_REL_PATH,
            "DeepNet (TFLite) 1-thread,full-inference with IREE-Dylib @ Unknown (CPU-ARMv8-A).json"
        ),
        os.path.join(
            self.tmp_dir.name, BENCHMARK_RESULTS_REL_PATH,
            "DeepNetv2 [f32] (TFLite) full-inference with IREE-Dylib-Sync @ Unknown (CPU-ARMv8-A).json"
        )
    ])
    self.assertEqual(driver.get_capture_filenames(), [
        os.path.join(
            self.tmp_dir.name, CAPTURES_REL_PATH,
            "DeepNet (TFLite) 1-thread,full-inference with IREE-Dylib @ Unknown (CPU-ARMv8-A).tracy"
        ),
        os.path.join(
            self.tmp_dir.name, CAPTURES_REL_PATH,
            "DeepNetv2 [f32] (TFLite) full-inference with IREE-Dylib-Sync @ Unknown (CPU-ARMv8-A).tracy"
        )
    ])
    self.assertEqual(driver.get_benchmark_errors(), [])

  def test_run_with_no_capture(self):
    self.config.trace_capture_config = None
    driver = FakeBenchmarkDriver(self.device_info, self.config,
                                 self.benchmark_suite)

    driver.run()

    self.assertEqual(len(driver.get_benchmark_result_filenames()), 2)
    self.assertEqual(driver.get_capture_filenames(), [])

  def test_run_with_exception_and_keep_going(self):
    self.config.keep_going = True
    driver = FakeBenchmarkDriver(self.device_info,
                                 self.config,
                                 self.benchmark_suite,
                                 raise_exception_on_case="case1")

    driver.run()

    self.assertEqual(len(driver.get_benchmark_errors()), 1)
    self.assertEqual(len(driver.get_benchmark_result_filenames()), 1)


if __name__ == "__main__":
  unittest.main()
