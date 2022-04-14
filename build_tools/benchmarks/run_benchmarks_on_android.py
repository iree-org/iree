#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Runs all matched benchmark suites on an Android device.

This script probes the Android phone via `adb` and uses the device information
to filter and run suitable benchmarks and optionally captures Tracy traces on
the Android phone.

It expects that `adb` is installed, and there is iree tools cross-compiled
towards Android. If to capture traces, another set of tracing-enabled iree
tools and the Tracy `capture` tool should be cross-compiled towards Android.

Example usages:

  # Without trace generation
  python3 run_benchmarks.py \
    --normal_benchmark_tool_dir=/path/to/normal/android/target/iree/tools/dir \
    /path/to/host/build/dir

  # With trace generation
  python3 run_benchmarks.py \
    --normal_benchmark_tool_dir=/path/to/normal/android/target/iree/tools/dir \
    --traced_benchmark_tool_dir=/path/to/tracy/android/target/iree/tools/dir \
    --trace_capture_tool=/path/to/host/build/tracy/capture \
    /path/to/host/build/dir
"""

import atexit
import os
import re
import subprocess
import tarfile
import shutil
import sys

from typing import Optional, Sequence
from common.benchmark_config import BenchmarkConfig
from common.benchmark_driver import BenchmarkDriver
from common.benchmark_definition import (execute_cmd,
                                         execute_cmd_and_get_output)
from common.benchmark_suite import (BenchmarkCase, BenchmarkSuite)
from common.android_device_utils import (get_android_device_model,
                                         get_android_device_info,
                                         get_android_gpu_name)
from common.common_arguments import build_common_argument_parser

# The flagfile/toolfile's filename for compiled benchmark artifacts.
MODEL_FLAGFILE_NAME = "flagfile"
MODEL_TOOLFILE_NAME = "tool"

# Root directory to perform benchmarks in on the Android device.
ANDROID_TMP_DIR = "/data/local/tmp/iree-benchmarks"

NORMAL_TOOL_REL_DIR = "normal-tools"
TRACED_TOOL_REL_DIR = "traced-tools"


def get_benchmark_repetition_count(runner: str) -> int:
  """Returns the benchmark repetition count for the given runner."""
  if runner == "iree-vmvx":
    # VMVX is very unoptimized for now and can take a long time to run.
    # Decrease the repetition for it until it's reasonably fast.
    return 3
  return 10


def get_git_commit_hash(commit: str) -> str:
  return execute_cmd_and_get_output(['git', 'rev-parse', commit],
                                    cwd=os.path.dirname(
                                        os.path.realpath(__file__)))


def adb_push_to_tmp_dir(content: str,
                        relative_dir: str = "",
                        verbose: bool = False) -> str:
  """Pushes content onto the Android device.

  Args:
    content: the full path to the source file.
    relative_dir: the directory to push to; relative to ANDROID_TMP_DIR.

  Returns:
    The full path to the content on the Android device.
  """
  filename = os.path.basename(content)
  android_path = os.path.join(ANDROID_TMP_DIR, relative_dir, filename)
  # When the output is a TTY, keep the default progress info output.
  # In other cases, redirect progress info to null to avoid bloating log files.
  stdout_redirect = None if sys.stdout.isatty() else subprocess.DEVNULL
  execute_cmd(
      ["adb", "push", os.path.abspath(content), android_path],
      verbose=verbose,
      stdout=stdout_redirect)
  return android_path


def adb_execute_and_get_output(cmd_args: Sequence[str],
                               relative_dir: str = "",
                               verbose: bool = False) -> str:
  """Executes command with adb shell.

  Switches to `relative_dir` relative to the android tmp directory before
  executing. Waits for completion and returns the command stdout.

  Args:
    cmd_args: a list containing the command to execute and its parameters
    relative_dir: the directory to execute the command in; relative to
      ANDROID_TMP_DIR.

  Returns:
    A string for the command output.
  """
  cmd = ["adb", "shell"]
  cmd.extend(["cd", os.path.join(ANDROID_TMP_DIR, relative_dir)])
  cmd.append("&&")
  cmd.extend(cmd_args)

  return execute_cmd_and_get_output(cmd, verbose=verbose)


def adb_execute(cmd_args: Sequence[str],
                relative_dir: str = "",
                verbose: bool = False) -> subprocess.CompletedProcess:
  """Executes command with adb shell.

  Switches to `relative_dir` relative to the android tmp directory before
  executing. Waits for completion. Output is streamed to the terminal.

  Args:
    cmd_args: a list containing the command to execute and its parameters
    relative_dir: the directory to execute the command in; relative to
      ANDROID_TMP_DIR.

  Returns:
    The completed process.
  """
  cmd = ["adb", "shell"]
  cmd.extend(["cd", os.path.join(ANDROID_TMP_DIR, relative_dir)])
  cmd.append("&&")
  cmd.extend(cmd_args)

  return execute_cmd(cmd, verbose=verbose)


def is_magisk_su():
  """Returns true if the Android device has a Magisk SU binary."""
  return "MagiskSU" in adb_execute_and_get_output(["su", "--help"])


def adb_execute_as_root(cmd_args: Sequence[str]) -> subprocess.CompletedProcess:
  """Executes the given command as root."""
  cmd = ["su", "-c" if is_magisk_su() else "root"]
  cmd.extend(cmd_args)
  return adb_execute(cmd)


def adb_start_cmd(cmd_args: Sequence[str],
                  relative_dir: str,
                  verbose: bool = False) -> subprocess.Popen:
  """Executes command with adb shell in a directory and returns the handle
  without waiting for completion.

  Args:
    cmd_args: a list containing the command to execute and its parameters
    relative_dir: the directory to execute the command in; relative to
      ANDROID_TMP_DIR.

  Returns:
    A Popen object for the started command.
  """
  cmd = ["adb", "shell"]
  cmd.extend(["cd", f"{ANDROID_TMP_DIR}/{relative_dir}"])
  cmd.append("&&")
  cmd.extend(cmd_args)

  if verbose:
    cmd_str = " ".join(cmd)
    print(f"cmd: {cmd_str}")
  return subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)


def get_vmfb_full_path_for_benchmark_case(benchmark_case_dir: str) -> str:
  flagfile_path = os.path.join(benchmark_case_dir, MODEL_FLAGFILE_NAME)
  flagfile = open(flagfile_path, "r")
  flagfile_lines = flagfile.readlines()
  for line in flagfile_lines:
    flag_name, flag_value = line.strip().split("=")
    if flag_name == "--module_file":
      # Realpath canonicalization matters. The caller may rely on that to track
      # which files it already pushed.
      return os.path.realpath(os.path.join(benchmark_case_dir, flag_value))
  raise ValueError(f"{flagfile_path} does not contain a --module_file flag")


class AndroidBenchmarkDriver(BenchmarkDriver):
  """Android benchmark driver."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.already_pushed_files = {}

  def run_benchmark_case(self, benchmark_case: BenchmarkCase,
                         benchmark_results_filename: Optional[str],
                         capture_filename: Optional[str]) -> None:
    benchmark_case_dir = benchmark_case.benchmark_case_dir
    android_case_dir = os.path.relpath(benchmark_case_dir,
                                       self.config.root_benchmark_dir)

    self.__push_vmfb_file(benchmark_case_dir)
    self.__check_and_push_file(
        os.path.join(benchmark_case_dir, MODEL_FLAGFILE_NAME), android_case_dir)

    taskset = self.__deduce_taskset(benchmark_case.bench_mode)

    if benchmark_results_filename is not None:
      self.__run_benchmark(android_case_dir=android_case_dir,
                           tool_name=benchmark_case.benchmark_tool_name,
                           driver=benchmark_case.driver,
                           results_filename=benchmark_results_filename,
                           taskset=taskset)

    if capture_filename is not None:
      self.__run_capture(android_case_dir=android_case_dir,
                         tool_name=benchmark_case.benchmark_tool_name,
                         capture_filename=capture_filename,
                         taskset=taskset)

  def __run_benchmark(self, android_case_dir: str, tool_name: str, driver: str,
                      results_filename: str, taskset: str):
    host_tool_path = os.path.join(self.config.normal_benchmark_tool_dir,
                                  tool_name)
    android_tool = self.__check_and_push_file(host_tool_path,
                                              NORMAL_TOOL_REL_DIR)
    cmd = [
        "taskset", taskset, android_tool, f"--flagfile={MODEL_FLAGFILE_NAME}"
    ]
    if tool_name == "iree-benchmark-module":
      cmd.extend([
          "--benchmark_format=json",
          "--benchmark_out_format=json",
          f"--benchmark_out='{os.path.basename(results_filename)}'",
      ])
      if self.config.benchmark_min_time:
        cmd.extend([
            f"--benchmark_min_time={self.config.benchmark_min_time}",
        ])
      else:
        repetitions = get_benchmark_repetition_count(driver)
        cmd.extend([
            f"--benchmark_repetitions={repetitions}",
        ])

    result_json = adb_execute_and_get_output(cmd,
                                             android_case_dir,
                                             verbose=self.verbose)

    # Pull the result file back onto the host and set the filename for later
    # return.
    pull_cmd = [
        "adb", "pull",
        os.path.join(ANDROID_TMP_DIR, android_case_dir,
                     os.path.basename(results_filename)), results_filename
    ]
    execute_cmd_and_get_output(pull_cmd, verbose=self.verbose)

    if self.verbose:
      print(result_json)

  def __run_capture(self, android_case_dir: str, tool_name: str,
                    capture_filename: str, taskset: str):
    capture_config = self.config.trace_capture_config
    host_tool_path = os.path.join(capture_config.traced_benchmark_tool_dir,
                                  tool_name)
    android_tool = self.__check_and_push_file(host_tool_path,
                                              TRACED_TOOL_REL_DIR)
    run_cmd = [
        "TRACY_NO_EXIT=1", f"IREE_PRESERVE_DYLIB_TEMP_FILES={ANDROID_TMP_DIR}",
        "taskset", taskset, android_tool, f"--flagfile={MODEL_FLAGFILE_NAME}"
    ]

    # Just launch the traced benchmark tool with TRACY_NO_EXIT=1 without
    # waiting for the adb command to complete as that won't happen.
    process = adb_start_cmd(run_cmd, android_case_dir, verbose=self.verbose)
    # But we do need to wait for its start; otherwise will see connection
    # failure when opening the catpure tool. Here we cannot just sleep a
    # certain amount of seconds---Pixel 4 seems to have an issue that will
    # make the trace collection step get stuck. Instead wait for the
    # benchmark result to be available.
    while True:
      line = process.stdout.readline()  # pytype: disable=attribute-error
      if line == "" and process.poll() is not None:  # Process completed
        raise ValueError("Cannot find benchmark result line in the log!")
      if self.verbose:
        print(line.strip())
      # Result available
      if re.match(r"^BM_.+/real_time", line) is not None:
        break

    # Now it's okay to collect the trace via the capture tool. This will
    # send the signal to let the previously waiting benchmark tool to
    # complete.
    capture_cmd = [
        capture_config.trace_capture_tool, "-f", "-o", capture_filename
    ]
    # If verbose, just let the subprocess print its output. The subprocess
    # may need to detect if the output is a TTY to decide whether to log
    # verbose progress info and use ANSI colors, so it's better to use
    # stdout redirection than to capture the output in a string.
    stdout_redirect = None if self.verbose else subprocess.DEVNULL
    execute_cmd(capture_cmd, verbose=self.verbose, stdout=stdout_redirect)

  def __deduce_taskset(self, bench_mode: Sequence[str]) -> str:
    """Deduces the CPU affinity taskset mask according to benchmark modes."""
    # TODO: we actually should check the number of cores the phone have.
    if "big-core" in bench_mode:
      return "80" if "1-thread" in bench_mode else "f0"
    if "little-core" in bench_mode:
      return "08" if "1-thread" in bench_mode else "0f"
    # Not specified: use the 7th core.
    return "80"

  def __push_vmfb_file(self, benchmark_case_dir: str):
    vmfb_path = get_vmfb_full_path_for_benchmark_case(benchmark_case_dir)
    vmfb_dir = os.path.dirname(vmfb_path)
    vmfb_rel_dir = os.path.relpath(vmfb_dir, self.config.root_benchmark_dir)
    self.__check_and_push_file(vmfb_path, vmfb_rel_dir)

  def __check_and_push_file(self, host_path: str, relative_dir: str):
    """Checks if the file has been pushed and pushes it if not."""
    android_path = self.already_pushed_files.get(host_path)
    if android_path is not None:
      return android_path

    android_path = adb_push_to_tmp_dir(host_path,
                                       relative_dir=relative_dir,
                                       verbose=self.verbose)
    self.already_pushed_files[host_path] = android_path
    return android_path


def set_cpu_frequency_scaling_governor(governor: str):
  git_root = execute_cmd_and_get_output(["git", "rev-parse", "--show-toplevel"])
  cpu_script = os.path.join(git_root, "build_tools", "benchmarks",
                            "set_android_scaling_governor.sh")
  android_path = adb_push_to_tmp_dir(cpu_script)
  adb_execute_as_root([android_path, governor])


def set_gpu_frequency_scaling_policy(policy: str):
  git_root = execute_cmd_and_get_output(["git", "rev-parse", "--show-toplevel"])
  device_model = get_android_device_model()
  gpu_name = get_android_gpu_name()
  if device_model == "Pixel-6" or device_model == "Pixel-6-Pro":
    gpu_script = os.path.join(git_root, "build_tools", "benchmarks",
                              "set_pixel6_gpu_scaling_policy.sh")
  elif gpu_name.lower().startswith("adreno"):
    gpu_script = os.path.join(git_root, "build_tools", "benchmarks",
                              "set_adreno_gpu_scaling_policy.sh")
  else:
    raise RuntimeError(
        f"Unsupported device '{device_model}' for setting GPU scaling policy")
  android_path = adb_push_to_tmp_dir(gpu_script)
  adb_execute_as_root([android_path, policy])


def main(args):
  device_info = get_android_device_info(args.verbose)
  if args.verbose:
    print(device_info)

  commit = get_git_commit_hash("HEAD")
  benchmark_config = BenchmarkConfig.build_from_args(args, commit)
  benchmark_suite = BenchmarkSuite.load_from_benchmark_suite_dir(
      benchmark_config.root_benchmark_dir)
  benchmark_driver = AndroidBenchmarkDriver(device_info=device_info,
                                            benchmark_config=benchmark_config,
                                            benchmark_suite=benchmark_suite,
                                            benchmark_grace_time=1.0,
                                            verbose=args.verbose)

  if args.continue_from_directory:
    # Merge in previous benchmarks and captures.
    benchmark_driver.add_previous_benchmarks_and_captures(
        args.continue_from_directory)

  if args.pin_cpu_freq:
    set_cpu_frequency_scaling_governor("performance")
    atexit.register(set_cpu_frequency_scaling_governor, "schedutil")
  if args.pin_gpu_freq:
    set_gpu_frequency_scaling_policy("performance")
    atexit.register(set_gpu_frequency_scaling_policy, "default")

  # Clear the benchmark directory on the Android device first just in case
  # there are leftovers from manual or failed runs.
  execute_cmd_and_get_output(["adb", "shell", "rm", "-rf", ANDROID_TMP_DIR],
                             verbose=args.verbose)

  if not args.no_clean:
    # Clear the benchmark directory on the Android device.
    atexit.register(execute_cmd_and_get_output,
                    ["adb", "shell", "rm", "-rf", ANDROID_TMP_DIR],
                    verbose=args.verbose)
    # Also clear temporary directory on the host device.
    atexit.register(shutil.rmtree, args.tmp_dir)

  # Tracy client and server communicate over port 8086 by default. If we want
  # to capture traces along the way, forward port via adb.
  trace_capture_config = benchmark_config.trace_capture_config
  if trace_capture_config:
    execute_cmd_and_get_output(["adb", "forward", "tcp:8086", "tcp:8086"],
                               verbose=args.verbose)
    atexit.register(execute_cmd_and_get_output,
                    ["adb", "forward", "--remove", "tcp:8086"],
                    verbose=args.verbose)

  benchmark_driver.run()

  benchmark_results = benchmark_driver.get_benchmark_results()
  if args.output is not None:
    with open(args.output, "w") as f:
      f.write(benchmark_results.to_json_str())

  if args.verbose:
    print(benchmark_results.commit)
    print(benchmark_results.benchmarks)

  if trace_capture_config:
    # Put all captures in a tarball and remove the origial files.
    with tarfile.open(trace_capture_config.capture_tarball, "w:gz") as tar:
      for capture_filename in benchmark_driver.get_capture_filenames():
        tar.add(capture_filename)

  benchmark_errors = benchmark_driver.get_benchmark_errors()
  if benchmark_errors:
    print("Benchmarking completed with errors", file=sys.stderr)
    raise RuntimeError(benchmark_errors)


if __name__ == "__main__":
  args = build_common_argument_parser().parse_args()
  main(args)
