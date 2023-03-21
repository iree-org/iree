# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import pathlib
import time
from typing import List, Optional, Sequence, Set, Tuple
from common.benchmark_suite import BenchmarkCase, BenchmarkSuite
from common.benchmark_config import BenchmarkConfig
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
    self.finished_benchmarks: List[Tuple[BenchmarkInfo, pathlib.Path]] = []
    self.finished_captures: List[pathlib.Path] = []
    self.benchmark_errors = []
    self._seen_benchmark_names: Set[str] = set()

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

  def run(self) -> None:
    """Execute the benchmark flow.

    It performs the following steps:
      1. Enumerate all categories in the benchmark suites.
      2. For each category, enumerate and filter benchmark cases.
      3. Call 'run_benchmark_case' for each benchmark case.
      4. Collect the benchmark results and captures.
    """

    self.config.benchmark_results_dir.mkdir(parents=True, exist_ok=True)
    if self.config.trace_capture_config is not None:
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
        benchmark_info = self.__get_benchmark_info_from_case(
            category=category, benchmark_case=benchmark_case)
        benchmark_name = str(benchmark_info)

        # Sanity check for the uniqueness of benchmark names.
        if benchmark_name in self._seen_benchmark_names:
          raise ValueError(
              f"Found duplicate benchmark {benchmark_name} in the suites.")
        self._seen_benchmark_names.add(benchmark_name)

        results_path, capture_path = self.__get_output_paths(benchmark_name)
        # If we continue from the previous results, check and skip if the result
        # files exist.
        if self.config.continue_from_previous:
          if results_path is not None and results_path.exists():
            self.finished_benchmarks.append((benchmark_info, results_path))
            results_path = None

          if capture_path is not None and capture_path.exists():
            self.finished_captures.append(capture_path)
            capture_path = None

        # Skip if no need to benchmark and capture.
        if results_path is None and capture_path is None:
          continue

        print(f"--> Benchmark started: {benchmark_name} <--")

        try:
          self.run_benchmark_case(benchmark_case, results_path, capture_path)
        except Exception as e:
          # Delete unfinished results if they exist.
          # TODO(#11087): Use missing_ok=True once we move to Python 3.8.
          if results_path is not None and results_path.is_file():
            results_path.unlink()
          if capture_path is not None and capture_path.is_file():
            capture_path.unlink()

          if not self.config.keep_going:
            raise e

          print(f"Processing of benchmark failed with: {e}")
          self.benchmark_errors.append(e)
          continue
        finally:
          # Some grace time.
          time.sleep(self.benchmark_grace_time)

        print("Benchmark completed")

        if results_path:
          self.finished_benchmarks.append((benchmark_info, results_path))
        if capture_path:
          self.finished_captures.append(capture_path)

  def get_benchmark_results(self) -> BenchmarkResults:
    """Returns the finished benchmark results."""

    results = BenchmarkResults()
    results.set_commit(self.config.git_commit_hash)

    finished_benchmarks = sorted(self.finished_benchmarks,
                                 key=lambda pair: str(pair[0]))
    for benchmark_info, path in finished_benchmarks:
      result_json_object = json.loads(path.read_text())
      benchmark_run = BenchmarkRun(benchmark_info,
                                   result_json_object["context"],
                                   result_json_object["benchmarks"])
      results.benchmarks.append(benchmark_run)

    return results

  def get_benchmark_result_filenames(self) -> Sequence[pathlib.Path]:
    """Returns the json file paths of finished benchmarks."""
    return list(path for _, path in self.finished_benchmarks)

  def get_capture_filenames(self) -> Sequence[pathlib.Path]:
    """Returns the tracy file paths of finished captures."""
    return self.finished_captures

  def get_benchmark_errors(self):
    """Returns the exceptions captured during benchmarking."""
    return self.benchmark_errors

  def __get_output_paths(self, benchmark_name: str):
    """Get output paths for the results and capture. The path of results/capture
    is None if the benchmark/capture doesn't need to be run.
    """

    benchmark_results_filename = None
    if self.config.normal_benchmark_tool_dir:
      benchmark_results_filename = self.config.benchmark_results_dir / f"{benchmark_name}.json"

    capture_filename = None
    if self.config.trace_capture_config:
      capture_filename = self.config.trace_capture_config.capture_tmp_dir / f"{benchmark_name}.tracy"

    return (benchmark_results_filename, capture_filename)

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
