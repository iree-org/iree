# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import pathlib
import tempfile
import unittest

from typing import Optional
from common.benchmark_suite import BenchmarkCase, BenchmarkSuite
from common.benchmark_driver import BenchmarkDriver
from common.benchmark_config import BENCHMARK_RESULTS_REL_PATH, CAPTURES_REL_PATH, BenchmarkConfig, TraceCaptureConfig
from common.benchmark_definition import IREE_DRIVERS_INFOS, DeviceInfo, PlatformType


class FakeBenchmarkDriver(BenchmarkDriver):

  def __init__(self,
               *args,
               raise_exception_on_case: Optional[str] = None,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.raise_exception_on_case = raise_exception_on_case

  def run_benchmark_case(self, benchmark_case: BenchmarkCase,
                         benchmark_results_filename: Optional[pathlib.Path],
                         capture_filename: Optional[pathlib.Path]) -> None:
    if (self.raise_exception_on_case is not None and
        self.raise_exception_on_case in str(benchmark_case.benchmark_case_dir)):
      raise Exception("fake exception")

    if benchmark_results_filename:
      benchmark_results_filename.write_text(
          json.dumps({
              "context": "fake_context",
              "benchmarks": [],
          }))
    if capture_filename:
      capture_filename.write_text("{}")


class BenchmarkDriverTest(unittest.TestCase):

  def setUp(self):
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.root_dir = tempfile.TemporaryDirectory()

    self.tmp_dir_path = pathlib.Path(self.tmp_dir.name)
    (self.tmp_dir_path / "build_config.txt").write_text(
        "IREE_HAL_DRIVER_LOCAL_SYNC=ON\n"
        "IREE_HAL_DRIVER_LOCAL_TASK=ON\n"
        "IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON\n")

    self.config = BenchmarkConfig(
        root_benchmark_dir=pathlib.Path(self.root_dir.name),
        benchmark_results_dir=self.tmp_dir_path / BENCHMARK_RESULTS_REL_PATH,
        git_commit_hash="abcd",
        normal_benchmark_tool_dir=self.tmp_dir_path,
        trace_capture_config=TraceCaptureConfig(
            traced_benchmark_tool_dir=self.tmp_dir_path,
            trace_capture_tool=self.tmp_dir_path / "capture_tool",
            capture_tarball=self.tmp_dir_path / "captures.tar",
            capture_tmp_dir=self.tmp_dir_path / CAPTURES_REL_PATH))

    self.device_info = DeviceInfo(platform_type=PlatformType.LINUX,
                                  model="Unknown",
                                  cpu_abi="arm64-v8a",
                                  cpu_uarch=None,
                                  cpu_features=["sha2"],
                                  gpu_name="Mali-G78")

    case1 = BenchmarkCase(model_name="DeepNet",
                          model_tags=[],
                          bench_mode=["1-thread", "full-inference"],
                          target_arch="CPU-ARM64-v8A",
                          driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu"],
                          benchmark_case_dir=pathlib.Path("case1"),
                          benchmark_tool_name="tool")
    case2 = BenchmarkCase(model_name="DeepNetv2",
                          model_tags=["f32"],
                          bench_mode=["full-inference"],
                          target_arch="CPU-ARM64-v8A",
                          driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu-sync"],
                          benchmark_case_dir=pathlib.Path("case2"),
                          benchmark_tool_name="tool")
    self.benchmark_suite = BenchmarkSuite({
        pathlib.Path("suite/TFLite"): [case1, case2],
    })

  def tearDown(self) -> None:
    self.root_dir.cleanup()
    self.tmp_dir.cleanup()

  def test_add_previous_benchmarks_and_captures(self):
    driver = BenchmarkDriver(self.device_info, self.config,
                             self.benchmark_suite)
    (self.tmp_dir_path / BENCHMARK_RESULTS_REL_PATH).mkdir(parents=True)
    (self.tmp_dir_path / CAPTURES_REL_PATH).mkdir(parents=True)
    benchmark_filename = (
        self.tmp_dir_path / BENCHMARK_RESULTS_REL_PATH /
        "MobileNetv2 [fp32,imagenet] (TFLite) big-core,full-inference with IREE-LLVM-CPU @ Pixel-4 (CPU-ARMv8.2-A).json"
    )
    benchmark_filename.touch()
    capture_filename = (
        self.tmp_dir_path / CAPTURES_REL_PATH /
        "MobileNetv2 [fp32,imagenet] (TFLite) big-core,full-inference with IREE-LLVM-CPU @ Pixel-4 (CPU-ARMv8.2-A).tracy"
    )
    capture_filename.touch()

    driver.add_previous_benchmarks_and_captures(self.tmp_dir_path)

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
        self.tmp_dir_path / BENCHMARK_RESULTS_REL_PATH /
        "DeepNet (TFLite) 1-thread,full-inference with IREE-LLVM-CPU @ Unknown (CPU-ARMv8-A).json",
        self.tmp_dir_path / BENCHMARK_RESULTS_REL_PATH /
        "DeepNetv2 [f32] (TFLite) full-inference with IREE-LLVM-CPU-Sync @ Unknown (CPU-ARMv8-A).json"
    ])
    self.assertEqual(driver.get_capture_filenames(), [
        self.tmp_dir_path / CAPTURES_REL_PATH /
        "DeepNet (TFLite) 1-thread,full-inference with IREE-LLVM-CPU @ Unknown (CPU-ARMv8-A).tracy",
        self.tmp_dir_path / CAPTURES_REL_PATH /
        "DeepNetv2 [f32] (TFLite) full-inference with IREE-LLVM-CPU-Sync @ Unknown (CPU-ARMv8-A).tracy"
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
