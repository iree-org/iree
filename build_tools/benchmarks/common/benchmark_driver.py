# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import pathlib
import time
from typing import Dict, Optional, Sequence, Tuple
from common.benchmark_suite import BenchmarkCase, BenchmarkSuite
from common.benchmark_config import BENCHMARK_RESULTS_REL_PATH, CAPTURES_REL_PATH, BenchmarkConfig
from common.benchmark_definition import BenchmarkInfo, BenchmarkResults, BenchmarkRun, DeviceInfo


class BenchmarkDriver(object):
  """Abstract driver runs the whole benchmark flow."""

  def __init__(self,
               device_info: DeviceInfo,
               benchmark_config: BenchmarkConfig,
               benchmark_suite: BenchmarkSuite,
               benchmark_grace_time: float = 0.0,
               verbose: bool = False):
    self.device_info = device_info
    self.config = benchmark_config
    self.benchmark_suite = benchmark_suite
    self.benchmark_grace_time = benchmark_grace_time
    self.verbose = verbose
    self.finished_benchmarks: Dict[str, Tuple[BenchmarkInfo, pathlib.Path]] = {}
    self.finished_captures: Dict[str, Tuple[BenchmarkInfo, pathlib.Path]] = {}
    self.benchmark_errors = []

  def run_benchmark_case(self, benchmark_case: BenchmarkCase,
                         benchmark_results_filename: Optional[pathlib.Path],
                         capture_filename: Optional[pathlib.Path]) -> None:
    """Runs the benchmark case and returns the results.

    Args:
      benchmark_case: the benchmark_case.
      benchmark_results_filename: the path to store benchmark results.
        Benchmarking is required if set.
      capture_filename: the path to store captured trace. Trace capturing is
        required if set.

    Raises:
      Exception during benchmarking.
    """
    raise NotImplementedError("Should be overwritten by a subclass.")

  def add_previous_benchmarks_and_captures(
      self, previous_directory: pathlib.Path) -> None:
    """Collect names of previous benchmarks and captures that should be skipped
    and merged into the results.
    """

    def get_key_value_pair(path: pathlib.Path):
      name = path.stem
      info = BenchmarkInfo.from_device_info_and_name(self.device_info, name)
      return (str(info), (info, path))

    previous_benchmark_filenames = set()
    previous_capture_filenames = set()
    previous_benchmarks_dir = previous_directory / BENCHMARK_RESULTS_REL_PATH
    if previous_benchmarks_dir.is_dir():
      previous_benchmark_filenames = set(
          previous_benchmarks_dir / p
          for p in previous_benchmarks_dir.iterdir()
          if p.suffix == ".json")

    previous_captures_dir = previous_directory / CAPTURES_REL_PATH
    if previous_captures_dir.is_dir():
      previous_capture_filenames = set(previous_captures_dir / p
                                       for p in previous_captures_dir.iterdir()
                                       if p.suffix == ".tracy")

    self.finished_benchmarks.update(
        get_key_value_pair(p) for p in previous_benchmark_filenames)
    self.finished_captures.update(
        get_key_value_pair(p) for p in previous_capture_filenames)

  def run(self) -> None:
    """Execute the benchmark flow.

    It performs the following steps:
      1. Enumerate all categories in the benchmark suites.
      2. For each category, enumerate and filter benchmark cases.
      3. Call 'run_benchmark_case' for each benchmark case.
      4. Collect the benchmark results and captures.
    """

    do_capture = self.config.trace_capture_config is not None

    self.config.benchmark_results_dir.mkdir(parents=True, exist_ok=True)
    if do_capture:
      self.config.trace_capture_config.capture_tmp_dir.mkdir(parents=True,
                                                             exist_ok=True)

    cpu_target_arch = self.device_info.get_iree_cpu_arch_name()
    gpu_target_arch = self.device_info.get_iree_gpu_arch_name()
    drivers, loaders = self.__get_available_drivers_and_loaders()

    for category, _ in self.benchmark_suite.list_categories():
      benchmark_cases = self.benchmark_suite.filter_benchmarks_for_category(
          category=category,
          available_drivers=drivers,
          available_loaders=loaders,
          cpu_target_arch_filter=f"^{cpu_target_arch}$",
          gpu_target_arch_filter=f"^{gpu_target_arch}$",
          driver_filter=self.config.driver_filter,
          mode_filter=self.config.mode_filter,
          model_name_filter=self.config.model_name_filter)

      for benchmark_case in benchmark_cases:
        (benchmark_info, benchmark_results_filename,
         capture_filename) = self.__get_benchmark_info_and_output_paths(
             category, benchmark_case)

        # Skip if no need to benchmark and capture.
        if not benchmark_results_filename and not capture_filename:
          continue

        benchmark_key = str(benchmark_info)
        print(f"--> Benchmark started: {benchmark_key} <--")

        try:
          self.run_benchmark_case(benchmark_case, benchmark_results_filename,
                                  capture_filename)
        except Exception as e:
          if not self.config.keep_going:
            raise e

          print(f"Processing of benchmark failed with: {e}")
          self.benchmark_errors.append(e)
          continue
        finally:
          # Some grace time.
          time.sleep(self.benchmark_grace_time)

        print("Benchmark completed")

        if benchmark_results_filename:
          self.finished_benchmarks[benchmark_key] = (benchmark_info,
                                                     benchmark_results_filename)
        if capture_filename:
          self.finished_captures[benchmark_key] = (benchmark_info,
                                                   capture_filename)

  def get_benchmark_results(self) -> BenchmarkResults:
    """Returns the finished benchmark results."""

    results = BenchmarkResults()
    results.set_commit(self.config.git_commit_hash)

    finished_benchmarks = list(self.finished_benchmarks.items())
    finished_benchmarks.sort(key=lambda b: b[0])

    for _, value in finished_benchmarks:
      benchmark_info, path = value
      with open(path) as f:
        result_json_object = json.loads(f.read())
      benchmark_run = BenchmarkRun(benchmark_info,
                                   result_json_object["context"],
                                   result_json_object["benchmarks"])
      results.benchmarks.append(benchmark_run)

    return results

  def get_benchmark_result_filenames(self) -> Sequence[pathlib.Path]:
    """Returns the json file paths of finished benchmarks."""
    return list(path for _, path in self.finished_benchmarks.values())

  def get_capture_filenames(self) -> Sequence[pathlib.Path]:
    """Returns the tracy file paths of finished captures."""
    return list(path for _, path in self.finished_captures.values())

  def get_benchmark_errors(self):
    """Returns the exceptions captured during benchmarking."""
    return self.benchmark_errors

  def __get_benchmark_info_and_output_paths(self, category: str,
                                            benchmark_case: BenchmarkCase):
    """Get benchmark info and paths for the results and capture. The path of
    results/capture is None if the benchmark/capture doesn't need to be run.
    """
    benchmark_info = self.__get_benchmark_info_from_case(
        category=category, benchmark_case=benchmark_case)
    benchmark_name = str(benchmark_info)

    benchmark_results_filename = None
    if (benchmark_name not in self.finished_benchmarks and
        self.config.normal_benchmark_tool_dir):
      benchmark_results_filename = self.config.benchmark_results_dir / f"{benchmark_name}.json"

    capture_filename = None
    if (benchmark_name not in self.finished_captures and
        self.config.trace_capture_config):
      capture_filename = self.config.trace_capture_config.capture_tmp_dir / f"{benchmark_name}.tracy"

    return (benchmark_info, benchmark_results_filename, capture_filename)

  def __get_benchmark_info_from_case(
      self, category: str, benchmark_case: BenchmarkCase) -> BenchmarkInfo:
    if benchmark_case.run_config is None:
      # TODO(#11076): Remove legacy path.
      return BenchmarkInfo(model_name=benchmark_case.model_name,
                           model_tags=benchmark_case.model_tags,
                           model_source=category,
                           bench_mode=benchmark_case.bench_mode,
                           driver_info=benchmark_case.driver_info,
                           device_info=self.device_info)

    run_tags = benchmark_case.run_config.module_execution_config.tags
    compile_tags = benchmark_case.run_config.module_generation_config.compile_config.tags
    return BenchmarkInfo(model_name=benchmark_case.model_name,
                         model_tags=benchmark_case.model_tags,
                         model_source=category,
                         bench_mode=run_tags,
                         compile_tags=compile_tags,
                         driver_info=benchmark_case.driver_info,
                         device_info=self.device_info,
                         run_config_id=benchmark_case.run_config.composite_id)

  def __get_available_drivers_and_loaders(
      self) -> Tuple[Sequence[str], Sequence[str]]:
    any_tool_dir = (self.config.normal_benchmark_tool_dir
                    if self.config.normal_benchmark_tool_dir else
                    self.config.trace_capture_config.traced_benchmark_tool_dir)
    config_txt_file_path = any_tool_dir / "build_config.txt"
    config_txt_file_lines = config_txt_file_path.read_text().splitlines()

    available_drivers = []
    available_loaders = []
    for line in config_txt_file_lines:
      name, value = line.strip().split("=")
      if value != "ON":
        continue
      if name == "IREE_HAL_DRIVER_CUDA":
        available_drivers.append("cuda")
      elif name == "IREE_HAL_DRIVER_LOCAL_SYNC":
        available_drivers.append("local-sync")
      elif name == "IREE_HAL_DRIVER_LOCAL_TASK":
        available_drivers.append("local-task")
      elif name == "IREE_HAL_DRIVER_VULKAN":
        available_drivers.append("vulkan")
      elif name == "IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF":
        available_loaders.append("embedded-elf")
      elif name == "IREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY":
        available_loaders.append("system-library")
      elif name == "IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE":
        available_loaders.append("vmvx-module")
      else:
        continue

    if self.verbose:
      available_drivers_str = ', '.join(available_drivers)
      print(f"Available drivers: {available_drivers_str}")
      available_loaders_str = ', '.join(available_loaders)
      print(f"Available loaders: {available_loaders_str}")

    return available_drivers, available_loaders
