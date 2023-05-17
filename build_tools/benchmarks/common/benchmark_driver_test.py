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
                                         BenchmarkMemory, BenchmarkMetrics)
from e2e_test_framework.definitions import common_definitions, iree_definitions


class FakeBenchmarkDriver(BenchmarkDriver):

  def __init__(self,
               *args,
               raise_exception_on_case: Optional[BenchmarkCase] = None,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.raise_exception_on_case = raise_exception_on_case
    self.run_benchmark_cases = []

  def run_benchmark_case(self, benchmark_case: BenchmarkCase,
                         benchmark_results_filename: Optional[pathlib.Path],
                         capture_filename: Optional[pathlib.Path]) -> None:
    if self.raise_exception_on_case == benchmark_case:
      raise Exception("fake exception")

    self.run_benchmark_cases.append(benchmark_case)

    if benchmark_results_filename:
      fake_benchmark_metrics = BenchmarkMetrics(
          real_time=BenchmarkLatency(0, 0, 0, "ns"),
          cpu_time=BenchmarkLatency(0, 0, 0, "ns"),
          host_memory=BenchmarkMemory(0, 0, 0, 0, "bytes"),
          device_memory=BenchmarkMemory(0, 0, 0, 0, "bytes"),
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
            capture_tmp_dir=self.captures_dir),
        use_compatible_filter=True)

    self.device_info = DeviceInfo(platform_type=PlatformType.LINUX,
                                  model="Unknown",
                                  cpu_abi="x86_64",
                                  cpu_uarch="CascadeLake",
                                  cpu_features=[],
                                  gpu_name="unknown")

    model_tflite = common_definitions.Model(
        id="tflite",
        name="model_tflite",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="",
        entry_function="predict",
        input_types=["1xf32"])
    device_spec = common_definitions.DeviceSpec.build(
        id="dev",
        device_name="test_dev",
        architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
        device_parameters=[],
        tags=[])
    compile_target = iree_definitions.CompileTarget(
        target_backend=iree_definitions.TargetBackend.LLVM_CPU,
        target_architecture=(
            common_definitions.DeviceArchitecture.X86_64_CASCADELAKE),
        target_abi=iree_definitions.TargetABI.LINUX_GNU)
    gen_config = iree_definitions.ModuleGenerationConfig.build(
        imported_model=iree_definitions.ImportedModel.from_model(model_tflite),
        compile_config=iree_definitions.CompileConfig.build(
            id="comp_a", tags=[], compile_targets=[compile_target]))
    exec_config_a = iree_definitions.ModuleExecutionConfig.build(
        id="exec_a",
        tags=["sync"],
        loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
        driver=iree_definitions.RuntimeDriver.LOCAL_SYNC)
    run_config_a = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=gen_config,
        module_execution_config=exec_config_a,
        target_device_spec=device_spec,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE)
    exec_config_b = iree_definitions.ModuleExecutionConfig.build(
        id="exec_b",
        tags=["task"],
        loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
        driver=iree_definitions.RuntimeDriver.LOCAL_TASK)
    run_config_b = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=gen_config,
        module_execution_config=exec_config_b,
        target_device_spec=device_spec,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE)
    self.case1 = BenchmarkCase(
        model_name="model_tflite",
        model_tags=[],
        bench_mode=["sync"],
        target_arch="x86_64-cascadelake",
        driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu-sync"],
        benchmark_case_dir=pathlib.Path("case1"),
        benchmark_tool_name="tool",
        run_config=run_config_a)
    self.case2 = BenchmarkCase(model_name="model_tflite",
                               model_tags=[],
                               bench_mode=["task"],
                               target_arch="x86_64-cascadelake",
                               driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu"],
                               benchmark_case_dir=pathlib.Path("case2"),
                               benchmark_tool_name="tool",
                               run_config=run_config_b)

    compile_target_rv64 = iree_definitions.CompileTarget(
        target_backend=iree_definitions.TargetBackend.LLVM_CPU,
        target_architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
        target_abi=iree_definitions.TargetABI.LINUX_GNU)
    gen_config_rv64 = iree_definitions.ModuleGenerationConfig.build(
        imported_model=iree_definitions.ImportedModel.from_model(model_tflite),
        compile_config=iree_definitions.CompileConfig.build(
            id="comp_rv64", tags=[], compile_targets=[compile_target_rv64]))
    device_spec_rv64 = common_definitions.DeviceSpec.build(
        id="rv64_dev",
        device_name="rv64_dev",
        architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
        device_parameters=[],
        tags=[])
    run_config_incompatible = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=gen_config_rv64,
        module_execution_config=exec_config_b,
        target_device_spec=device_spec_rv64,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE)
    self.incompatible_case = BenchmarkCase(
        model_name="model_tflite",
        model_tags=[],
        bench_mode=["task"],
        target_arch="riscv_64-generic",
        driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu"],
        benchmark_case_dir=pathlib.Path("incompatible_case"),
        benchmark_tool_name="tool",
        run_config=run_config_incompatible)
    self.benchmark_suite = BenchmarkSuite({
        pathlib.Path("suite/TFLite"): [
            self.case1, self.case2, self.incompatible_case
        ],
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
        self.benchmark_results_dir / f"{self.case1.run_config}.json",
        self.benchmark_results_dir / f"{self.case2.run_config}.json"
    ])
    self.assertEqual(driver.get_capture_filenames(), [
        self.captures_dir / f"{self.case1.run_config}.tracy",
        self.captures_dir / f"{self.case2.run_config}.tracy"
    ])
    self.assertEqual(driver.get_benchmark_errors(), [])

  def test_run_disable_compatible_filter(self):
    self.config.use_compatible_filter = False
    driver = FakeBenchmarkDriver(self.device_info, self.config,
                                 self.benchmark_suite)

    driver.run()

    self.assertEqual(len(driver.get_benchmark_results().benchmarks), 3)

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
                                 raise_exception_on_case=self.case1)

    driver.run()

    self.assertEqual(len(driver.get_benchmark_errors()), 1)
    self.assertEqual(len(driver.get_benchmark_result_filenames()), 1)

  def test_run_with_previous_benchmarks_and_captures(self):
    benchmark_filename = (self.benchmark_results_dir /
                          f"{self.case1.run_config}.json")
    benchmark_filename.touch()
    capture_filename = self.captures_dir / f"{self.case1.run_config}.tracy"
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
