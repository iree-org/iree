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

import argparse
import atexit
import json
import os
import re
import subprocess
import tarfile
import time
import shutil
import sys

from typing import Any, Dict, List, Optional, Sequence, Tuple, TextIO, Set

from common.benchmark_definition import (
    AndroidDeviceInfo, BenchmarkInfo, BenchmarkResults, BenchmarkRun,
    execute_cmd, execute_cmd_and_get_output, get_android_device_model,
    get_android_gpu_name, IREE_PRETTY_NAMES_TO_DRIVERS)
from common.benchmark_suite import (BENCHMARK_SUITE_REL_PATH,
                                    compose_info_object,
                                    filter_benchmarks_for_category)

# The flagfile/toolfile's filename for compiled benchmark artifacts.
MODEL_FLAGFILE_NAME = "flagfile"
MODEL_TOOLFILE_NAME = "tool"

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
    device_info: AndroidDeviceInfo,
    root_benchmark_dir: str,
    benchmark_category_dir: str,
    benchmark_case_dirs: Sequence[str],
    tmp_dir: str,
    normal_benchmark_tool_dir: str,
    traced_benchmark_tool_dir: Optional[str] = None,
    trace_capture_tool: Optional[str] = None,
    skip_benchmarks: Optional[Set[str]] = None,
    skip_captures: Optional[Set[str]] = None,
    do_capture: bool = False,
    keep_going: bool = False,
    benchmark_min_time: float = 0,
    verbose: bool = False,
) -> Tuple[Sequence[Tuple[Optional[str], Optional[str]]], Sequence[Exception]]:
  """Runs all benchmarks on the Android device and reports results and captures.

  Args:
    device_info: an AndroidDeviceInfo object.
    root_benchmark_dir: path to the benchmark suite within the root build dir
    benchmark_category_dir: the directory to a specific benchmark category.
    benchmark_case_dirs: a list of benchmark case directories.
    tmp_dir: path to temporary directory to which intermediate outputs should be
      stored. Separate "benchmark-results" and "captures" subdirectories will be
      created as necessary.
    normal_benchmark_tool_dir: the path to the normal benchmark tool directory.
    traced_benchmark_tool_dir: the path to the tracing-enabled benchmark tool
      directory.
    trace_capture_tool: the path to the tool for collecting captured traces.
    skip_benchmarks: names of benchmarks that should be skipped. Note that
      captures will still be run for these benchmarks if do_capture is true and
      they are not also in skip_captures.
    skip_captures: names of benchmark captures that should be skipped.
    do_capture: whether captures should be collected.
    keep_going: whether to proceed if an individual run fails. Exceptions will
      logged and returned.
    benchmark_min_time: min number of seconds to run the benchmark for, if
      specified. Otherwise, the benchmark will be repeated a fixed number of
      times.
    verbose: whether to print additional debug information.

  Returns:
    A tuple with a list containing (benchmark-filename, capture-filename) tuples
    and a list containing raised exceptions (only if keep_going is true)
  """
  push_vmfb_files(
      benchmark_case_dirs=benchmark_case_dirs,
      root_benchmark_dir=root_benchmark_dir,
      verbose=verbose,
  )

  # Create directories on the host to store results from each benchmark run.
  benchmark_results_dir = os.path.join(tmp_dir, "benchmark-results")
  os.makedirs(benchmark_results_dir, exist_ok=True)

  # And the same for captures, if we are collecting them.
  captures_dir = os.path.join(tmp_dir, "captures")
  if do_capture:
    os.makedirs(captures_dir, exist_ok=True)

  results = []
  errors = []
  skip_benchmarks = skip_benchmarks if skip_benchmarks else set()
  skip_captures = skip_captures if skip_captures else set()

  # Push all model artifacts to the device and run them.
  root_benchmark_dir = os.path.dirname(benchmark_category_dir)
  for benchmark_case_dir in benchmark_case_dirs:
    # Read the file specifying which tool should be used for benchmarking
    with open(os.path.join(benchmark_case_dir, MODEL_TOOLFILE_NAME)) as f:
      tool = f.read().strip()
      adb_push_to_tmp_dir(os.path.join(normal_benchmark_tool_dir, tool),
                          relative_dir=NORMAL_TOOL_REL_DIR,
                          verbose=verbose)
      if do_capture:
        adb_push_to_tmp_dir(os.path.join(traced_benchmark_tool_dir, tool),
                            relative_dir=TRACED_TOOL_REL_DIR,
                            verbose=verbose)

    benchmark_info = compose_info_object(device_info, benchmark_category_dir,
                                         benchmark_case_dir)
    benchmark_key = str(benchmark_info)
    # If we're not running the benchmark or the capture, just skip ahead.
    # No need to push files.
    if (benchmark_key in skip_benchmarks) and (not do_capture or
                                               benchmark_key in skip_captures):
      continue

    print(f"--> benchmark: {benchmark_info} <--")
    # Now try to actually run benchmarks and collect captures. If keep_going is
    # True then errors in the underlying commands will be logged and returned.
    try:
      android_relative_dir = os.path.relpath(benchmark_case_dir,
                                             root_benchmark_dir)
      adb_push_to_tmp_dir(os.path.join(benchmark_case_dir, MODEL_FLAGFILE_NAME),
                          android_relative_dir,
                          verbose=verbose)

      benchmark_result_filename = None
      if benchmark_key not in skip_benchmarks:
        benchmark_results_basename = f"{benchmark_key}.json"

        cmd = [
            "taskset",
            benchmark_info.deduce_taskset(),
            os.path.join(ANDROID_TMP_DIR, NORMAL_TOOL_REL_DIR, tool),
            f"--flagfile={MODEL_FLAGFILE_NAME}"
        ]
        if tool == "iree-benchmark-module":
          cmd.extend([
              "--benchmark_format=json",
              "--benchmark_out_format=json",
              f"--benchmark_out='{benchmark_results_basename}'",
          ])
          if benchmark_min_time:
            cmd.extend([
                f"--benchmark_min_time={benchmark_min_time}",
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
        benchmark_result_filename = os.path.join(benchmark_results_dir,
                                                 benchmark_results_basename)
        pull_cmd = [
            "adb", "pull",
            os.path.join(ANDROID_TMP_DIR, android_relative_dir,
                         benchmark_results_basename), benchmark_result_filename
        ]
        execute_cmd_and_get_output(pull_cmd, verbose=verbose)

        if verbose:
          print(result_json)

      capture_filename = None
      if do_capture and benchmark_key not in skip_captures:
        run_cmd = [
            "TRACY_NO_EXIT=1", "taskset",
            benchmark_info.deduce_taskset(),
            os.path.join(ANDROID_TMP_DIR, TRACED_TOOL_REL_DIR, tool),
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
        capture_filename = os.path.join(captures_dir, f"{benchmark_key}.tracy")
        capture_cmd = [trace_capture_tool, "-f", "-o", capture_filename]
        capture_log = execute_cmd_and_get_output(capture_cmd, verbose=verbose)
        if verbose:
          print(capture_log)

      print("...benchmark completed")

      results.append((benchmark_result_filename, capture_filename))
      time.sleep(1)  # Some grace time.

    except subprocess.CalledProcessError as e:
      if keep_going:
        print(f"Processing of benchmark failed with: {e}")
        errors.append(e)
        continue
      raise e

  return (results, errors)


def get_available_drivers(tool_dir: str, verbose: bool) -> Sequence[str]:
  config_txt_file_path = os.path.join(tool_dir, "build_config.txt")
  config_txt_file = open(config_txt_file_path, "r")
  config_txt_file_lines = config_txt_file.readlines()
  available_drivers = []
  for line in config_txt_file_lines:
    name, value = line.strip().split("=")
    if value != "ON":
      continue
    if name == "IREE_HAL_DRIVER_CUDA":
      available_drivers.append("cuda")
    elif name == "IREE_HAL_DRIVER_DYLIB":
      available_drivers.append("dylib")
    elif name == "IREE_HAL_DRIVER_DYLIB_SYNC":
      available_drivers.append("dylib-sync")
    elif name == "IREE_HAL_DRIVER_EXPERIMENTAL_ROCM":
      available_drivers.append("rocm")
    elif name == "IREE_HAL_DRIVER_VMVX":
      available_drivers.append("vmvx")
    elif name == "IREE_HAL_DRIVER_VMVX_SYNC":
      available_drivers.append("vmvx-sync")
    elif name == "IREE_HAL_DRIVER_VULKAN":
      available_drivers.append("vulkan")
    else:
      continue
  if verbose:
    available_drivers_str = ', '.join(available_drivers)
    print(f"Available drivers: {available_drivers_str}")
  return available_drivers


def filter_and_run_benchmarks(
    device_info: AndroidDeviceInfo,
    root_build_dir: str,
    driver_filter: Optional[str],
    model_name_filter: Optional[str],
    mode_filter: Optional[str],
    tmp_dir: str,
    normal_benchmark_tool_dir: str,
    traced_benchmark_tool_dir: Optional[str],
    trace_capture_tool: Optional[str],
    skip_benchmarks: Optional[Set[str]],
    skip_captures: Optional[Set[str]],
    do_capture: bool = False,
    keep_going: bool = False,
    benchmark_min_time: float = 0,
    verbose: bool = False) -> Tuple[List[str], List[str], List[Exception]]:
  """Filters and runs benchmarks in all categories for the given device.

  Args:
    device_info: an AndroidDeviceInfo object.
    root_build_dir: the root build directory containing the built benchmark
      suites.
    driver_filter: filter benchmarks to those whose driver matches this regex
      (or all if this is None).
    model_name_filter: filter benchmarks to those whose model name matches this
      regex (or all if this is None).
    mode_filter: filter benchmarks to those whose benchmarking mode matches this
      regex (or all if this is None).
    tmp_dir: path to temporary directory to which intermediate outputs should be
      stored. Separate "benchmark-results" and "captures" subdirectories will be
      created as necessary.
    normal_benchmark_tool_dir: the path to the normal benchmark tool directory.
    traced_benchmark_tool_dir: the path to the tracing-enabled benchmark tool
      directory.
    trace_capture_tool: the path to the tool for collecting captured traces.
    skip_benchmarks: names of benchmarks that should be skipped. Note that
      captures will still be run for these benchmarks if do_capture is true and
      they are not also in skip_captures.
    skip_captures: names of benchmark captures that should be skipped.
    do_capture: whether captures should be collected.
    keep_going: whether to proceed if an individual run fails. Exceptions will
      logged and returned.
    benchmark_min_time: min number of seconds to run the benchmark for, if
      specified. Otherwise, the benchmark will be repeated a fixed number of
      times.
    verbose: whether to print additional debug information.

  Returns:
    Lists of benchmark file paths, capture file paths, and exceptions raise
    (only if keep_going is True).
  """
  cpu_target_arch = CPU_ABI_TO_TARGET_ARCH_MAP[device_info.cpu_abi.lower()]
  gpu_target_arch = GPU_NAME_TO_TARGET_ARCH_MAP[device_info.gpu_name.lower()]

  root_benchmark_dir = os.path.join(root_build_dir, BENCHMARK_SUITE_REL_PATH)

  benchmark_files = []
  captures = []
  errors = []

  skip_benchmarks = skip_benchmarks if skip_benchmarks else set()

  for directory in sorted(os.listdir(root_benchmark_dir)):
    benchmark_category_dir = os.path.join(root_benchmark_dir, directory)
    available_drivers = get_available_drivers(
        tool_dir=normal_benchmark_tool_dir, verbose=verbose)
    matched_benchmarks = filter_benchmarks_for_category(
        benchmark_category_dir=benchmark_category_dir,
        cpu_target_arch_filter=cpu_target_arch,
        gpu_target_arch_filter=gpu_target_arch,
        driver_filter=driver_filter,
        model_name_filter=model_name_filter,
        mode_filter=mode_filter,
        available_drivers=available_drivers,
        verbose=verbose)
    run_results, run_errors = run_benchmarks_for_category(
        device_info=device_info,
        root_benchmark_dir=root_benchmark_dir,
        benchmark_category_dir=benchmark_category_dir,
        benchmark_case_dirs=matched_benchmarks,
        tmp_dir=tmp_dir,
        normal_benchmark_tool_dir=normal_benchmark_tool_dir,
        traced_benchmark_tool_dir=traced_benchmark_tool_dir,
        skip_benchmarks=skip_benchmarks,
        trace_capture_tool=trace_capture_tool,
        do_capture=do_capture,
        keep_going=keep_going,
        benchmark_min_time=benchmark_min_time,
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


def parse_arguments():
  """Parses command-line options."""

  def check_dir_path(path):
    if os.path.isdir(path):
      return path
    else:
      raise argparse.ArgumentTypeError(path)

  def check_exe_path(path):
    if os.access(path, os.X_OK):
      return path
    else:
      raise argparse.ArgumentTypeError(f"'{path}' is not an executable")

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "build_dir",
      metavar="<build-dir>",
      type=check_dir_path,
      help="Path to the build directory containing benchmark suites")
  parser.add_argument("--normal_benchmark_tool_dir",
                      "--normal-benchmark-tool-dir",
                      type=check_exe_path,
                      required=True,
                      help="Path to the normal iree tool directory")
  parser.add_argument("--traced_benchmark_tool_dir",
                      "--traced-benchmark-tool-dir",
                      type=check_exe_path,
                      default=None,
                      help="Path to the tracing-enabled iree tool directory")
  parser.add_argument("--trace_capture_tool",
                      "--trace-capture-tool",
                      type=check_exe_path,
                      default=None,
                      help="Path to the tool for collecting captured traces")
  parser.add_argument(
      "--driver-filter-regex",
      "--driver_filter_regex",
      type=str,
      default=None,
      help="Only run benchmarks matching the given driver regex")
  parser.add_argument(
      "--model-name-regex",
      "--model_name_regex",
      type=str,
      default=None,
      help="Only run benchmarks matching the given model name regex")
  parser.add_argument(
      "--mode-regex",
      "--mode_regex",
      type=str,
      default=None,
      help="Only run benchmarks matching the given benchmarking mode regex")
  parser.add_argument("--output",
                      "-o",
                      default=None,
                      help="Path to the output file")
  parser.add_argument("--capture_tarball",
                      "--capture-tarball",
                      default=None,
                      help="Path to the tarball for captures")
  parser.add_argument("--no-clean",
                      action="store_true",
                      help="Do not clean up the temporary directory used for "
                      "benchmarking on the Android device")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")
  parser.add_argument(
      "--pin-cpu-freq",
      "--pin_cpu_freq",
      action="store_true",
      help="Pin CPU frequency for all cores to the maximum. Requires root")
  parser.add_argument("--pin-gpu-freq",
                      "--pin_gpu_freq",
                      action="store_true",
                      help="Pin GPU frequency to the maximum. Requires root")
  parser.add_argument(
      "--keep_going",
      "--keep-going",
      action="store_true",
      help="Continue running after a failed benchmark. The overall exit status"
      " will still indicate failure and all errors will be reported at the end."
  )
  parser.add_argument(
      "--tmp_dir",
      "--tmp-dir",
      "--tmpdir",
      default="/tmp/iree-benchmarks",
      help="Base directory in which to store temporary files. A subdirectory"
      " with a name matching the git commit hash will be created.")
  parser.add_argument(
      "--continue_from_directory",
      "--continue-from-directory",
      default=None,
      help="Path to directory with previous benchmark temporary files. This"
      " should be for the specific commit (not the general tmp-dir). Previous"
      " benchmark and capture results from here will not be rerun and will be"
      " combined with the new runs.")
  parser.add_argument(
      "--benchmark_min_time",
      "--benchmark-min-time",
      default=0,
      type=float,
      help="If specified, this will be passed as --benchmark_min_time to the"
      "iree-benchmark-module (minimum number of seconds to repeat running "
      "for). In that case, no --benchmark_repetitions flag will be passed."
      " If not specified, a --benchmark_repetitions will be passed "
      "instead.")

  args = parser.parse_args()

  return args


def main(args):
  device_info = AndroidDeviceInfo.from_adb()
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

  previous_benchmarks = None
  previous_captures = None

  do_capture = (args.traced_benchmark_tool_dir is not None and
                args.trace_capture_tool is not None)

  # Collect names of previous benchmarks and captures that should be skipped and
  # merged into the results.
  if args.continue_from_directory is not None:
    previous_benchmarks_dir = os.path.join(args.continue_from_directory,
                                           "benchmark-results")
    if os.path.isdir(previous_benchmarks_dir):
      previous_benchmarks = set(
          os.path.splitext(os.path.basename(p))[0]
          for p in os.listdir(previous_benchmarks_dir))
    if do_capture:
      previous_captures_dir = os.path.join(args.continue_from_directory,
                                           "captures")
      if os.path.isdir(previous_captures_dir):
        previous_captures = set(
            os.path.splitext(os.path.basename(p))[0]
            for p in os.listdir(previous_captures_dir))

  # Clear the benchmark directory on the Android device first just in case
  # there are leftovers from manual or failed runs.
  execute_cmd_and_get_output(["adb", "shell", "rm", "-rf", ANDROID_TMP_DIR],
                             verbose=args.verbose)

  if not args.no_clean:
    # Clear the benchmark directory on the Android device.
    atexit.register(execute_cmd_and_get_output,
                    ["adb", "shell", "rm", "-rf", ANDROID_TMP_DIR],
                    verbose=args.verbose)

  # Tracy client and server communicate over port 8086 by default. If we want
  # to capture traces along the way, forward port via adb.
  if do_capture:
    execute_cmd_and_get_output(["adb", "forward", "tcp:8086", "tcp:8086"])
    atexit.register(execute_cmd_and_get_output,
                    ["adb", "forward", "--remove", "tcp:8086"])

    args.traced_benchmark_tool_dir = os.path.realpath(
        args.traced_benchmark_tool_dir)
    args.trace_capture_tool = os.path.realpath(args.trace_capture_tool)

  results = BenchmarkResults()
  commit = get_git_commit_hash("HEAD")
  results.set_commit(commit)

  args.tmp_dir = os.path.join(args.tmp_dir, commit)
  os.makedirs(args.tmp_dir, exist_ok=True)

  benchmarks, captures, errors = filter_and_run_benchmarks(
      device_info=device_info,
      root_build_dir=args.build_dir,
      driver_filter=args.driver_filter_regex,
      model_name_filter=args.model_name_regex,
      mode_filter=args.mode_regex,
      tmp_dir=args.tmp_dir,
      normal_benchmark_tool_dir=os.path.realpath(
          args.normal_benchmark_tool_dir),
      traced_benchmark_tool_dir=args.traced_benchmark_tool_dir,
      trace_capture_tool=args.trace_capture_tool,
      skip_benchmarks=previous_benchmarks,
      skip_captures=previous_captures,
      do_capture=do_capture,
      keep_going=args.keep_going,
      benchmark_min_time=args.benchmark_min_time,
      verbose=args.verbose)

  # Merge in previous benchmarks and captures.
  if previous_benchmarks:
    benchmarks.extend(f"{os.path.join(previous_benchmarks_dir, b)}.json"
                      for b in previous_benchmarks)
  if do_capture and previous_captures:
    captures.extend(f"{os.path.join(previous_captures_dir, c)}.tracy"
                    for c in previous_captures)

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

  # Delete all the temp files if everything completed successfully.
  if not args.no_clean and not errors:
    shutil.rmtree(args.tmp_dir)

  if errors:
    print("Benchmarking completed with errors", file=sys.stderr)
    raise RuntimeError(errors)


if __name__ == "__main__":
  main(parse_arguments())
