# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import dataclasses
import json
import pathlib
import tempfile
from typing import Optional
import unittest

from common import benchmark_config
from common.benchmark_suite import BenchmarkCase, BenchmarkSuite
from common.benchmark_driver import BenchmarkDriver
from common.benchmark_definition import (IREE_DRIVERS_INFOS, DeviceInfo,
                                         PlatformType, BenchmarkLatency,
                                         BenchmarkMetrics)


class FakeBenchmarkDriver(BenchmarkDriver):

  def __init__(self,
               *args,
               raise_exception_on_case: Optional[str] = None,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.raise_exception_on_case = raise_exception_on_case
    self.run_benchmark_cases = []

  def run_benchmark_case(self, benchmark_case: BenchmarkCase,
                         benchmark_results_filename: Optional[pathlib.Path],
                         capture_filename: Optional[pathlib.Path]) -> None:
    if (self.raise_exception_on_case is not None and
        self.raise_exception_on_case in str(benchmark_case.benchmark_case_dir)):
      raise Exception("fake exception")

    self.run_benchmark_cases.append(benchmark_case)

    if benchmark_results_filename:
      fake_benchmark_metrics = BenchmarkMetrics(
          real_time=BenchmarkLatency(0, 0, 0, "ns"),
          cpu_time=BenchmarkLatency(0, 0, 0, "ns"),
          raw_data={},
      )
      benchmark_results_filename.write_text(
          json.dumps(fake_benchmark_metrics.to_json_object()))
    if capture_filename:
      capture_filename.write_text("{}")


class BenchmarkDriverTest(unittest.TestCase):

  def setUp(self):
    self._tmp_dir_obj = tempfile.TemporaryDirectory()
    self._root_dir_obj = tempfile.TemporaryDirectory()

    self.tmp_dir = pathlib.Path(self._tmp_dir_obj.name)
    (self.tmp_dir / "build_config.txt").write_text(
        "IREE_HAL_DRIVER_LOCAL_SYNC=ON\n"
        "IREE_HAL_DRIVER_LOCAL_TASK=ON\n"
        "IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON\n")

    self.benchmark_results_dir = (self.tmp_dir /
                                  benchmark_config.BENCHMARK_RESULTS_REL_PATH)
    self.captures_dir = (self.tmp_dir / benchmark_config.CAPTURES_REL_PATH)
    self.benchmark_results_dir.mkdir()
    self.captures_dir.mkdir()

    self.config = benchmark_config.BenchmarkConfig(
        root_benchmark_dir=pathlib.Path(self._root_dir_obj.name),
        benchmark_results_dir=self.benchmark_results_dir,
        git_commit_hash="abcd",
        normal_benchmark_tool_dir=self.tmp_dir,
        trace_capture_config=benchmark_config.TraceCaptureConfig(
            traced_benchmark_tool_dir=self.tmp_dir,
            trace_capture_tool=self.tmp_dir / "capture_tool",
            capture_tarball=self.tmp_dir / "captures.tar",
            capture_tmp_dir=self.captures_dir))

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
    self._tmp_dir_obj.cleanup()
    self._root_dir_obj.cleanup()

  def test_run(self):
    driver = FakeBenchmarkDriver(self.device_info, self.config,
                                 self.benchmark_suite)

    driver.run()

    self.assertEqual(driver.get_benchmark_results().commit, "abcd")
    self.assertEqual(len(driver.get_benchmark_results().benchmarks), 2)
    self.assertEqual(
        driver.get_benchmark_results().benchmarks[0].metrics.raw_data, {})
    self.assertEqual(driver.get_benchmark_result_filenames(), [
        self.benchmark_results_dir /
        "DeepNet (TFLite) 1-thread,full-inference with IREE-LLVM-CPU @ Unknown (CPU-ARMv8-A).json",
        self.benchmark_results_dir /
        "DeepNetv2 [f32] (TFLite) full-inference with IREE-LLVM-CPU-Sync @ Unknown (CPU-ARMv8-A).json"
    ])
    self.assertEqual(driver.get_capture_filenames(), [
        self.captures_dir /
        "DeepNet (TFLite) 1-thread,full-inference with IREE-LLVM-CPU @ Unknown (CPU-ARMv8-A).tracy",
        self.captures_dir /
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

  def test_run_with_previous_benchmarks_and_captures(self):
    benchmark_filename = (
        self.benchmark_results_dir /
        "DeepNet (TFLite) 1-thread,full-inference with IREE-LLVM-CPU @ Unknown (CPU-ARMv8-A).json"
    )
    benchmark_filename.touch()
    capture_filename = (
        self.captures_dir /
        "DeepNet (TFLite) 1-thread,full-inference with IREE-LLVM-CPU @ Unknown (CPU-ARMv8-A).tracy"
    )
    capture_filename.touch()
    config = dataclasses.replace(self.config, continue_from_previous=True)
    driver = FakeBenchmarkDriver(device_info=self.device_info,
                                 benchmark_config=config,
                                 benchmark_suite=self.benchmark_suite)

    driver.run()

    self.assertEqual(len(driver.run_benchmark_cases), 1)
    self.assertEqual(len(driver.get_benchmark_result_filenames()), 2)
    self.assertEqual(len(driver.get_capture_filenames()), 2)


if __name__ == "__main__":
  unittest.main()
