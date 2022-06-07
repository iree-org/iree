#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Runs all matched benchmark suites on a Linux device."""

import subprocess
import atexit
import os
import re
import shutil
import sys
import tarfile

from typing import Optional

from common.benchmark_driver import BenchmarkDriver
from common.benchmark_suite import MODEL_FLAGFILE_NAME, BenchmarkCase, BenchmarkSuite
from common.benchmark_config import BenchmarkConfig
from common.benchmark_definition import execute_cmd, execute_cmd_and_get_output, get_git_commit_hash, get_iree_benchmark_module_arguments, wait_for_iree_benchmark_module_start
from common.common_arguments import build_common_argument_parser
from common.linux_device_utils import get_linux_device_info


class LinuxBenchmarkDriver(BenchmarkDriver):
  """Linux benchmark driver."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def run_benchmark_case(self, benchmark_case: BenchmarkCase,
                         benchmark_results_filename: Optional[str],
                         capture_filename: Optional[str]) -> None:

    # TODO(pzread): Taskset should be derived from CPU topology.
    # Only use the low 8 cores.
    taskset = "0xFF"

    if benchmark_results_filename:
      self.__run_benchmark(case_dir=benchmark_case.benchmark_case_dir,
                           tool_name=benchmark_case.benchmark_tool_name,
                           results_filename=benchmark_results_filename,
                           config=benchmark_case.config,
                           taskset=taskset)

    if capture_filename:
      self.__run_capture(case_dir=benchmark_case.benchmark_case_dir,
                         tool_name=benchmark_case.benchmark_tool_name,
                         capture_filename=capture_filename,
                         taskset=taskset)

  def __run_benchmark(self, case_dir, tool_name: str, results_filename: str,
                      config: str, taskset: str):
    tool_path = os.path.join(self.config.normal_benchmark_tool_dir, tool_name)
    cmd = ["taskset", taskset, tool_path, f"--flagfile={MODEL_FLAGFILE_NAME}"]
    if tool_name == "iree-benchmark-module":
      cmd.extend(
          get_iree_benchmark_module_arguments(
              results_filename=results_filename,
              config=config,
              benchmark_min_time=self.config.benchmark_min_time))

    result_json = execute_cmd_and_get_output(cmd,
                                             cwd=case_dir,
                                             verbose=self.verbose)
    if self.verbose:
      print(result_json)

  def __run_capture(self, case_dir, tool_name: str, capture_filename: str,
                    taskset: str):
    capture_config = self.config.trace_capture_config

    tool_path = os.path.join(capture_config.traced_benchmark_tool_dir,
                             tool_name)
    cmd = ["taskset", taskset, tool_path, f"--flagfile={MODEL_FLAGFILE_NAME}"]
    process = subprocess.Popen(cmd,
                               env={"TRACY_NO_EXIT": "1"},
                               cwd=case_dir,
                               stdout=subprocess.PIPE,
                               universal_newlines=True)

    wait_for_iree_benchmark_module_start(process, self.verbose)

    capture_cmd = [
        capture_config.trace_capture_tool, "-f", "-o", capture_filename
    ]
    stdout_redirect = None if self.verbose else subprocess.DEVNULL
    execute_cmd(capture_cmd, verbose=self.verbose, stdout=stdout_redirect)


def main(args):
  device_info = get_linux_device_info(args.device_model, args.cpu_uarch,
                                      args.verbose)
  if args.verbose:
    print(device_info)

  commit = get_git_commit_hash("HEAD")
  benchmark_config = BenchmarkConfig.build_from_args(args, commit)
  benchmark_suite = BenchmarkSuite.load_from_benchmark_suite_dir(
      benchmark_config.root_benchmark_dir)
  benchmark_driver = LinuxBenchmarkDriver(device_info=device_info,
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
    with open(args.output, "w") as f:
      f.write(benchmark_results.to_json_str())

  if args.verbose:
    print(benchmark_results.commit)
    print(benchmark_results.benchmarks)

  trace_capture_config = benchmark_config.trace_capture_config
  if trace_capture_config:
    # Put all captures in a tarball and remove the origial files.
    with tarfile.open(trace_capture_config.capture_tarball, "w:gz") as tar:
      for capture_filename in benchmark_driver.get_capture_filenames():
        tar.add(capture_filename)

  benchmark_errors = benchmark_driver.get_benchmark_errors()
  if benchmark_errors:
    print("Benchmarking completed with errors", file=sys.stderr)
    raise RuntimeError(benchmark_errors)


def parse_argument():
  arg_parser = build_common_argument_parser()
  arg_parser.add_argument("--device_model",
                          default="Unknown",
                          help="Device model")
  arg_parser.add_argument("--cpu_uarch",
                          default=None,
                          help="CPU microarchitecture, e.g., CascadeLake")

  return arg_parser.parse_args()


if __name__ == "__main__":
  main(parse_argument())
