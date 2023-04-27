#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Runs all matched benchmark suites on a Linux device."""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

from typing import Any, List, Optional
import typing
import atexit
import json
import shutil
import subprocess
import tarfile

from common.benchmark_driver import BenchmarkDriver
from common.benchmark_suite import (MODEL_FLAGFILE_NAME, BenchmarkCase,
                                    BenchmarkSuite)
from common.benchmark_config import BenchmarkConfig
from common.benchmark_definition import (execute_cmd,
                                         execute_cmd_and_get_output,
                                         get_git_commit_hash,
                                         get_iree_benchmark_module_arguments,
                                         wait_for_iree_benchmark_module_start,
                                         parse_iree_benchmark_metrics)
from common.linux_device_utils import get_linux_device_info
from e2e_test_framework.definitions import iree_definitions
from e2e_test_framework import serialization
from e2e_test_artifacts import iree_artifacts
from e2e_model_tests import run_module_utils
import common.common_arguments


class LinuxBenchmarkDriver(BenchmarkDriver):
  """Linux benchmark driver."""

  def __init__(self, gpu_id: str, *args, **kwargs):
    self.gpu_id = gpu_id
    super().__init__(*args, **kwargs)

  def run_benchmark_case(self, benchmark_case: BenchmarkCase,
                         benchmark_results_filename: Optional[pathlib.Path],
                         capture_filename: Optional[pathlib.Path]) -> None:

    if benchmark_results_filename:
      self.__run_benchmark(benchmark_case=benchmark_case,
                           results_filename=benchmark_results_filename)

    if capture_filename:
      self.__run_capture(benchmark_case=benchmark_case,
                         capture_filename=capture_filename)

  def __parse_flagfile(self, case_dir: pathlib.Path) -> List[str]:
    return [
        line.strip()
        for line in (case_dir / MODEL_FLAGFILE_NAME).read_text().splitlines()
    ]

  def __build_tool_cmds(self, benchmark_case: BenchmarkCase,
                        tool_path: pathlib.Path) -> List[Any]:
    run_config = benchmark_case.run_config
    if run_config is None:
      # TODO(#11076): Remove legacy path.
      if benchmark_case.benchmark_case_dir is None:
        raise ValueError(
            "benchmark_case_dir can't be None if run_config is None.")

      # TODO(pzread): Taskset should be derived from CPU topology.
      # Only use the low 8 cores.
      cmds = ["taskset", "0xFF", tool_path]

      run_flags = self.__parse_flagfile(benchmark_case.benchmark_case_dir)
      # Replace the CUDA device flag with the specified GPU.
      if benchmark_case.driver_info.driver_name == "cuda":
        run_flags = [
            flag for flag in run_flags if not flag.startswith("--device")
        ]
        run_flags.append(f"--device=cuda://{self.gpu_id}")

      return cmds + run_flags

    cmds: List[Any] = run_module_utils.build_linux_wrapper_cmds_for_device_spec(
        run_config.target_device_spec)
    cmds.append(tool_path)

    module_dir_path = iree_artifacts.get_module_dir_path(
        run_config.module_generation_config,
        root_path=self.config.root_benchmark_dir)
    cmds += [f"--module={module_dir_path / iree_artifacts.MODULE_FILENAME}"]
    cmds += run_config.materialize_run_flags(gpu_id=self.gpu_id)

    return cmds

  def __run_benchmark(self, benchmark_case: BenchmarkCase,
                      results_filename: pathlib.Path):
    if self.config.normal_benchmark_tool_dir is None:
      raise ValueError("normal_benchmark_tool_dir can't be None.")

    tool_name = benchmark_case.benchmark_tool_name
    tool_path = self.config.normal_benchmark_tool_dir / tool_name
    cmd = self.__build_tool_cmds(benchmark_case=benchmark_case,
                                 tool_path=tool_path)

    if tool_name == "iree-benchmark-module":
      cmd.extend(
          get_iree_benchmark_module_arguments(
              results_filename=str(results_filename),
              driver_info=benchmark_case.driver_info,
              benchmark_min_time=self.config.benchmark_min_time))

    benchmark_stdout = execute_cmd_and_get_output(
        cmd, cwd=benchmark_case.benchmark_case_dir, verbose=self.verbose)
    benchmark_metrics = parse_iree_benchmark_metrics(benchmark_stdout)
    if self.verbose:
      print(benchmark_metrics)
    results_filename.write_text(json.dumps(benchmark_metrics.to_json_object()))

  def __run_capture(self, benchmark_case: BenchmarkCase,
                    capture_filename: pathlib.Path):
    capture_config = self.config.trace_capture_config
    if capture_config is None:
      raise ValueError("capture_config can't be None.")

    tool_path = (capture_config.traced_benchmark_tool_dir /
                 benchmark_case.benchmark_tool_name)
    cmd = self.__build_tool_cmds(benchmark_case=benchmark_case,
                                 tool_path=tool_path)
    process = subprocess.Popen(cmd,
                               env={"TRACY_NO_EXIT": "1"},
                               cwd=benchmark_case.benchmark_case_dir,
                               stdout=subprocess.PIPE,
                               text=True)

    wait_for_iree_benchmark_module_start(process, self.verbose)

    capture_cmd = [
        capture_config.trace_capture_tool, "-f", "-o", capture_filename
    ]
    stdout_redirect = None if self.verbose else subprocess.DEVNULL
    execute_cmd(capture_cmd, verbose=self.verbose, stdout=stdout_redirect)


def main(args):
  device_info = get_linux_device_info(args.device_model, args.cpu_uarch,
                                      args.gpu_id, args.verbose)
  if args.verbose:
    print(device_info)

  commit = get_git_commit_hash("HEAD")
  benchmark_config = BenchmarkConfig.build_from_args(args, commit)

  if args.execution_benchmark_config is None:
    # TODO(#11076): Remove legacy path.
    benchmark_suite = BenchmarkSuite.load_from_benchmark_suite_dir(
        benchmark_config.root_benchmark_dir)
  else:
    benchmark_groups = json.loads(args.execution_benchmark_config.read_text())
    benchmark_group = benchmark_groups.get(args.target_device_name)
    if benchmark_group is None:
      raise ValueError("Target device not found in the benchmark config.")
    run_configs = serialization.unpack_and_deserialize(
        data=benchmark_group["run_configs"],
        root_type=typing.List[iree_definitions.E2EModelRunConfig])
    benchmark_suite = BenchmarkSuite.load_from_run_configs(
        run_configs=run_configs)

  benchmark_driver = LinuxBenchmarkDriver(gpu_id=args.gpu_id,
                                          device_info=device_info,
                                          benchmark_config=benchmark_config,
                                          benchmark_suite=benchmark_suite,
                                          benchmark_grace_time=1.0,
                                          verbose=args.verbose)

  if args.pin_cpu_freq:
    raise NotImplementedError("CPU freq pinning is not supported yet.")
  if args.pin_gpu_freq:
    raise NotImplementedError("GPU freq pinning is not supported yet.")
  if not args.no_clean:
    atexit.register(shutil.rmtree, args.tmp_dir)

  benchmark_driver.run()

  benchmark_results = benchmark_driver.get_benchmark_results()
  if args.output is not None:
    with args.output.open("w") as f:
      f.write(benchmark_results.to_json_str())

  if args.verbose:
    print(benchmark_results.commit)
    print(benchmark_results.benchmarks)

  trace_capture_config = benchmark_config.trace_capture_config
  if trace_capture_config:
    # Put all captures in a tarball and remove the original files.
    with tarfile.open(trace_capture_config.capture_tarball, "w:gz") as tar:
      for capture_filename in benchmark_driver.get_capture_filenames():
        tar.add(capture_filename)

  benchmark_errors = benchmark_driver.get_benchmark_errors()
  if benchmark_errors:
    print("Benchmarking completed with errors", file=sys.stderr)
    raise RuntimeError(benchmark_errors)


def parse_argument():
  arg_parser = common.common_arguments.Parser()
  arg_parser.add_argument("--device_model",
                          default="Unknown",
                          help="Device model")
  arg_parser.add_argument("--cpu_uarch",
                          default=None,
                          help="CPU microarchitecture, e.g., CascadeLake")
  arg_parser.add_argument(
      "--gpu_id",
      type=str,
      default="0",
      help="GPU ID to run the benchmark, e.g., '0' or 'GPU-<UUID>'")

  return arg_parser.parse_args()


if __name__ == "__main__":
  main(parse_argument())
