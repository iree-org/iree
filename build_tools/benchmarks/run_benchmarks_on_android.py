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
import json
import os
import re
import subprocess
import tarfile
import time
import shutil
import sys

from typing import List, Optional, Sequence, Tuple
from common.benchmark_suite import BENCHMARK_RESULTS_REL_PATH, CAPTURES_REL_PATH, MODEL_FLAGFILE_NAME, BenchmarkCase, BenchmarkConfig, BenchmarkHelper

from common.benchmark_definition import (DeviceInfo, BenchmarkInfo,
                                         BenchmarkResults, BenchmarkRun,
                                         execute_cmd,
                                         execute_cmd_and_get_output)
from common.android_device_utils import (get_android_device_model,
                                         get_android_device_info,
                                         get_android_gpu_name)
from common.common_arguments import build_common_argument_parser

# Root directory to perform benchmarks in on the Android device.
ANDROID_TMP_DIR = "/data/local/tmp/iree-benchmarks"

NORMAL_TOOL_REL_DIR = "normal-tools"
TRACED_TOOL_REL_DIR = "traced-tools"

# A map from Android CPU ABI to IREE's benchmark target architecture.
CPU_ABI_TO_TARGET_ARCH_MAP = {
    "arm64-v8a": "cpu-arm64-v8a",
}

# A map from Android GPU name to IREE's benchmark target architecture.
GPU_NAME_TO_TARGET_ARCH_MAP = {
    "adreno-640": "gpu-adreno",
    "adreno-650": "gpu-adreno",
    "adreno-660": "gpu-adreno",
    "adreno-730": "gpu-adreno",
    "mali-g77": "gpu-mali-valhall",
    "mali-g78": "gpu-mali-valhall",
}


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
  execute_cmd(["adb", "push", "-p",
               os.path.abspath(content), android_path],
              verbose=verbose)
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


def push_vmfb_files(benchmark_case_dirs: Sequence[str], root_benchmark_dir: str,
                    verbose: bool):
  vmfb_files_already_pushed = set()
  for case_dir in benchmark_case_dirs:
    vmfb_path = get_vmfb_full_path_for_benchmark_case(case_dir)
    if vmfb_path in vmfb_files_already_pushed:
      continue
    vmfb_dir = os.path.dirname(vmfb_path)
    vmfb_rel_dir = os.path.relpath(vmfb_dir, root_benchmark_dir)
    adb_push_to_tmp_dir(vmfb_path, relative_dir=vmfb_rel_dir, verbose=verbose)
    vmfb_files_already_pushed.add(vmfb_path)


def run_benchmarks_for_category(
    category: str,
    device_info: DeviceInfo,
    benchmark_config: BenchmarkConfig,
    benchmark_cases: Sequence[BenchmarkCase],
    verbose: bool = False,
) -> Tuple[Sequence[Tuple[Optional[str], Optional[str]]], Sequence[Exception]]:
  """Runs all benchmarks on the Android device and reports results and captures.

  Args:
    category: the benchmark category.
    device_info: an DeviceInfo object.
    benchmark_config: the benchmark config.
    benchmark_cases: a list of benchmark cases.
    verbose: whether to print additional debug information.

  Returns:
    A tuple with a list containing (benchmark-filename, capture-filename) tuples
    and a list containing raised exceptions (only if keep_going is true)
  """
  push_vmfb_files(
      benchmark_case_dirs=[
          benchmark_case.benchmark_case_dir
          for benchmark_case in benchmark_cases
      ],
      root_benchmark_dir=benchmark_config.root_benchmark_dir,
      verbose=verbose,
  )

  results = []
  errors = []

  # Push all model artifacts to the device and run them.
  for benchmark_case in benchmark_cases:
    if benchmark_case.normal_benchmark_tool_path:
      adb_push_to_tmp_dir(benchmark_case.normal_benchmark_tool_path,
                          relative_dir=NORMAL_TOOL_REL_DIR,
                          verbose=verbose)
    if benchmark_case.traced_benchmark_tool_path:
      adb_push_to_tmp_dir(benchmark_case.traced_benchmark_tool_path,
                          relative_dir=TRACED_TOOL_REL_DIR,
                          verbose=verbose)

    benchmark_info = benchmark_case.benchmark_info
    print(f"--> benchmark: {benchmark_info} <--")
    # Now try to actually run benchmarks and collect captures. If keep_going is
    # True then errors in the underlying commands will be logged and returned.
    benchmark_case_dir = benchmark_case.benchmark_case_dir
    try:
      android_relative_dir = os.path.relpath(
          benchmark_case_dir, benchmark_config.root_benchmark_dir)
      adb_push_to_tmp_dir(os.path.join(benchmark_case_dir, MODEL_FLAGFILE_NAME),
                          android_relative_dir,
                          verbose=verbose)

      benchmark_results_filename = None
      benchmark_tool = os.path.basename(
          benchmark_case.normal_benchmark_tool_path)
      if not benchmark_case.skip_normal_benchmark:
        benchmark_results_basename = os.path.basename(
            benchmark_case.benchmark_results_filename)
        cmd = [
            "taskset",
            benchmark_info.deduce_taskset(),
            os.path.join(ANDROID_TMP_DIR, NORMAL_TOOL_REL_DIR, benchmark_tool),
            f"--flagfile={MODEL_FLAGFILE_NAME}"
        ]
        if benchmark_tool == "iree-benchmark-module":
          cmd.extend([
              "--benchmark_format=json",
              "--benchmark_out_format=json",
              f"--benchmark_out='{benchmark_results_basename}'",
          ])
          if benchmark_config.benchmark_min_time:
            cmd.extend([
                f"--benchmark_min_time={benchmark_config.benchmark_min_time}",
            ])
          else:
            repetitions = get_benchmark_repetition_count(benchmark_info.runner)
            cmd.extend([
                f"--benchmark_repetitions={repetitions}",
            ])

        result_json = adb_execute_and_get_output(cmd,
                                                 android_relative_dir,
                                                 verbose=verbose)

        # Pull the result file back onto the host and set the filename for later
        # return.
        benchmark_results_filename = benchmark_case.benchmark_results_filename
        pull_cmd = [
            "adb", "pull",
            os.path.join(ANDROID_TMP_DIR, android_relative_dir,
                         benchmark_results_basename), benchmark_results_filename
        ]
        execute_cmd_and_get_output(pull_cmd, verbose=verbose)

        if verbose:
          print(result_json)

      capture_filename = None
      if not benchmark_case.skip_traced_benchmark:
        benchmark_tool = os.path.basename(
            benchmark_case.traced_benchmark_tool_path)
        run_cmd = [
            "TRACY_NO_EXIT=1",
            f"IREE_PRESERVE_DYLIB_TEMP_FILES={ANDROID_TMP_DIR}", "taskset",
            benchmark_info.deduce_taskset(),
            os.path.join(ANDROID_TMP_DIR, TRACED_TOOL_REL_DIR, benchmark_tool),
            f"--flagfile={MODEL_FLAGFILE_NAME}"
        ]

        # Just launch the traced benchmark tool with TRACY_NO_EXIT=1 without
        # waiting for the adb command to complete as that won't happen.
        process = adb_start_cmd(run_cmd, android_relative_dir, verbose=verbose)
        # But we do need to wait for its start; otherwise will see connection
        # failure when opening the catpure tool. Here we cannot just sleep a
        # certain amount of seconds---Pixel 4 seems to have an issue that will
        # make the trace collection step get stuck. Instead wait for the
        # benchmark result to be available.
        while True:
          line = process.stdout.readline()  # pytype: disable=attribute-error
          if line == "" and process.poll() is not None:  # Process completed
            raise ValueError("Cannot find benchmark result line in the log!")
          if verbose:
            print(line.strip())
          # Result available
          if re.match(r"^BM_.+/real_time", line) is not None:
            break

        # Now it's okay to collect the trace via the capture tool. This will
        # send the signal to let the previously waiting benchmark tool to
        # complete.
        capture_filename = benchmark_case.capture_filename
        capture_cmd = [
            benchmark_config.trace_capture_tool, "-f", "-o", capture_filename
        ]
        capture_log = execute_cmd_and_get_output(capture_cmd, verbose=verbose)
        if verbose:
          print(capture_log)

      print("...benchmark completed")

      results.append((benchmark_results_filename, capture_filename))
      time.sleep(1)  # Some grace time.

    except subprocess.CalledProcessError as e:
      if benchmark_config.keep_going:
        print(f"Processing of benchmark failed with: {e}")
        errors.append(e)
        continue
      raise e

  return (results, errors)


def filter_and_run_benchmarks(
    device_info: DeviceInfo,
    benchmark_config: BenchmarkConfig,
    verbose: bool = False) -> Tuple[List[str], List[str], List[Exception]]:
  """Filters and runs benchmarks in all categories for the given device.

  Args:
    device_info: an DeviceInfo object.
    benchmark_config: the benchmark config.
    verbose: whether to print additional debug information.

  Returns:
    Lists of benchmark file paths, capture file paths, and exceptions raise
    (only if keep_going is True).
  """
  cpu_target_arch = CPU_ABI_TO_TARGET_ARCH_MAP[device_info.cpu_abi.lower()]
  gpu_target_arch = GPU_NAME_TO_TARGET_ARCH_MAP[device_info.gpu_name.lower()]

  benchmark_helper = BenchmarkHelper(benchmark_config, device_info)
  benchmark_files = []
  captures = []
  errors = []

  # Create directories on the host to store results and captures from each benchmark run.
  os.makedirs(benchmark_config.benchmark_results_dir, exist_ok=True)
  if benchmark_config.do_capture:
    os.makedirs(benchmark_config.capture_dir, exist_ok=True)

  drivers = benchmark_helper.get_available_drivers(verbose)

  for category in benchmark_helper.list_benchmark_categories():
    benchmark_cases = benchmark_helper.generate_benchmark_cases(
        category, cpu_target_arch, gpu_target_arch, drivers, verbose)

    run_results, run_errors = run_benchmarks_for_category(
        category=category,
        device_info=device_info,
        benchmark_config=benchmark_config,
        benchmark_cases=benchmark_cases,
        verbose=verbose)
    errors.extend(run_errors)
    for benchmark_filename, capture_filename in run_results:
      if benchmark_filename is not None:
        benchmark_files.append(benchmark_filename)
      if capture_filename is not None:
        captures.append(capture_filename)

  return (benchmark_files, captures, errors)


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

  if device_info.cpu_abi.lower() not in CPU_ABI_TO_TARGET_ARCH_MAP:
    raise ValueError(f"Unrecognized CPU ABI: '{device_info.cpu_abi}'; "
                     "need to update the map")
  if device_info.gpu_name.lower() not in GPU_NAME_TO_TARGET_ARCH_MAP:
    raise ValueError(f"Unrecognized GPU name: '{device_info.gpu_name}'; "
                     "need to update the map")

  if args.pin_cpu_freq:
    set_cpu_frequency_scaling_governor("performance")
    atexit.register(set_cpu_frequency_scaling_governor, "schedutil")
  if args.pin_gpu_freq:
    set_gpu_frequency_scaling_policy("performance")
    atexit.register(set_gpu_frequency_scaling_policy, "default")

  previous_benchmarks = set()
  previous_captures = set()
  # Collect names of previous benchmarks and captures that should be skipped and
  # merged into the results.
  if args.continue_from_directory is not None:
    previous_benchmarks_dir = os.path.join(args.continue_from_directory,
                                           BENCHMARK_RESULTS_REL_PATH)
    if os.path.isdir(previous_benchmarks_dir):
      previous_benchmarks = set(
          os.path.splitext(os.path.basename(p))[0]
          for p in os.listdir(previous_benchmarks_dir))

    previous_captures_dir = os.path.join(args.continue_from_directory,
                                         CAPTURES_REL_PATH)
    if os.path.isdir(previous_captures_dir):
      previous_captures = set(
          os.path.splitext(os.path.basename(p))[0]
          for p in os.listdir(previous_captures_dir))

  commit = get_git_commit_hash("HEAD")
  benchmark_config = BenchmarkConfig.build(args=args,
                                           git_commit_hash=commit,
                                           skip_benchmarks=previous_benchmarks,
                                           skip_captures=previous_captures)

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
  if benchmark_config.do_capture:
    execute_cmd_and_get_output(["adb", "forward", "tcp:8086", "tcp:8086"],
                               verbose=args.verbose)
    atexit.register(execute_cmd_and_get_output,
                    ["adb", "forward", "--remove", "tcp:8086"],
                    verbose=args.verbose)

  os.makedirs(benchmark_config.tmp_dir, exist_ok=True)

  benchmarks, captures, errors = filter_and_run_benchmarks(
      device_info=device_info,
      benchmark_config=benchmark_config,
      verbose=args.verbose)

  # Merge in previous benchmarks and captures.
  if previous_benchmarks:
    benchmarks.extend(f"{os.path.join(previous_benchmarks_dir, b)}.json"
                      for b in previous_benchmarks)
  if benchmark_config.do_capture and previous_captures:
    captures.extend(f"{os.path.join(previous_captures_dir, c)}.tracy"
                    for c in previous_captures)

  results = BenchmarkResults()
  results.set_commit(commit)
  for b in benchmarks:
    with open(b) as f:
      result_json_object = json.loads(f.read())
    benchmark_info = BenchmarkInfo.from_device_info_and_name(
        device_info,
        os.path.splitext(os.path.basename(b))[0])
    benchmark_run = BenchmarkRun(benchmark_info, result_json_object["context"],
                                 result_json_object["benchmarks"])
    results.benchmarks.append(benchmark_run)

  if args.output is not None:
    with open(args.output, "w") as f:
      f.write(results.to_json_str())

  if args.verbose:
    print(results.commit)
    print(results.benchmarks)

  if captures:
    # Put all captures in a tarball and remove the origial files.
    with tarfile.open(args.capture_tarball, "w:gz") as tar:
      for capture_filename in captures:
        tar.add(capture_filename)

  if errors:
    print("Benchmarking completed with errors", file=sys.stderr)
    raise RuntimeError(errors)


if __name__ == "__main__":
  args = build_common_argument_parser().parse_args()
  main(args)
