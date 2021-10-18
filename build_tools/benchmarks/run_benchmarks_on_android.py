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

It expects that `adb` is installed, and there is an `iree-benchmark-module`
tool cross-compiled towards Android. If to capture traces, another
tracing-enabled `iree-benchmark-module` and the Tracy `capture` tool should be
cross-compiled towards Android.

It also expects the benchmark artifacts are generated by building the
`iree-benchmark-suites` target in the following directory structure:

<root-build-dir>/benchmark_suites
└── <benchmark-category> (e.g., TensorFlow)
    ├── <benchmark-suite> (e.g., MobileBertSquad-fp32)
    │   ├── <benchmark-case> (e.g., iree-vulkan__GPU-Mali-Valhall__kernel-execution)
    │   │   └── flagfile
    │   ├── ...
    │   │   └── flagfile
    │   └── <benchmark_case>
    │       └── flagfile
    └── vmfb
        ├── compiled-<sha1>.vmfb
        ├── ...
        └── compiled-<sha1>.vmfb

Example usages:

  # Without trace generation
  python3 run_benchmarks.py \
    --normal_benchmark_tool=/path/to/android/target/iree-benchmark_module \
    /path/to/host/build/dir

  # With trace generation
  python3 run_benchmarks.py \
    --normal_benchmark_tool=/path/to/normal/android/target/iree-benchmark_module \
    --traced_benchmark_tool=/path/to/tracy/android/target/iree-benchmark_module \
    --trace_capture_tool=/path/to/host/build/tracy/capture \
    /path/to/host/build/dir
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time

from typing import Any, Dict, Optional, Sequence, Tuple

from common.benchmark_definition import (AndroidDeviceInfo, BenchmarkInfo,
                                         BenchmarkResults,
                                         execute_cmd_and_get_output)

# All benchmarks' relative path against root build directory.
BENCHMARK_SUITE_REL_PATH = "benchmark_suites"
# VMFB files' relative path against a benchmark category directory.
VMFB_REL_PATH = "vmfb"

# The flagfile's filename for compiled benchmark artifacts.
MODEL_FLAGFILE_NAME = "flagfile"

# Root directory to perform benchmarks in on the Android device.
ANDROID_TMP_DIR = "/data/local/tmp/iree-benchmarks"

# A map from Android CPU ABI to IREE's benchmark target architecture.
CPU_ABI_TO_TARGET_ARCH_MAP = {
    "arm64-v8a": "cpu-arm64-v8a",
}

# A map from Android GPU name to IREE's benchmark target architecture.
GPU_NAME_TO_TARGET_ARCH_MAP = {
    "adreno-640": "gpu-adreno",
    "adreno-650": "gpu-adreno",
    "adreno-660": "gpu-adreno",
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
                        relative_dir: str,
                        verbose: bool = False) -> str:
  """Pushes content onto the Android device.

  Args:
  - content: the full path to the source file.
  - relative_dir: the directory to push to; relative to ANDROID_TMP_DIR.

  Returns:
  - The full path to the content on the Android device.
  """
  filename = os.path.basename(content)
  android_path = os.path.join(ANDROID_TMP_DIR, relative_dir, filename)
  execute_cmd_and_get_output(
      ["adb", "push", os.path.abspath(content), android_path], verbose=verbose)
  return android_path


def adb_execute_in_dir(cmd_args: Sequence[str],
                       relative_dir: str,
                       verbose: bool = False) -> str:
  """Executes command with adb shell in a directory, waits for completion,
  and returns the output.

  Args:
  - cmd_args: a list containing the command to execute and its parameters
  - relative_dir: the directory to execute the command in; relative to
    ANDROID_TMP_DIR.

  Returns:
  - A string for the command output.
  """
  cmd = ["adb", "shell"]
  cmd.extend(["cd", f"{ANDROID_TMP_DIR}/{relative_dir}"])
  cmd.append("&&")
  cmd.extend(cmd_args)

  return execute_cmd_and_get_output(cmd, verbose=verbose)


def adb_start_in_dir(cmd_args: Sequence[str],
                     relative_dir: str,
                     verbose: bool = False) -> subprocess.Popen:
  """Executes command with adb shell in a directory and returns the handle
  without waiting for completion.

  Args:
  - cmd_args: a list containing the command to execute and its parameters
  - relative_dir: the directory to execute the command in; relative to
    ANDROID_TMP_DIR.

  Returns:
  - A Popen object for the started command.
  """
  cmd = ["adb", "shell"]
  cmd.extend(["cd", f"{ANDROID_TMP_DIR}/{relative_dir}"])
  cmd.append("&&")
  cmd.extend(cmd_args)

  if verbose:
    cmd_str = " ".join(cmd)
    print(f"cmd: {cmd_str}")
  return subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)


def compose_benchmark_info_object(device_info: AndroidDeviceInfo,
                                  benchmark_category_dir: str,
                                  benchmark_case_dir: str) -> BenchmarkInfo:
  """Creates an BenchmarkInfo object to describe the benchmark.

  Args:
  - device_info: an AndroidDeviceInfo object.
  - benchmark_category_dir: the directory to a specific benchmark category.
  - benchmark_case_dir: a directory containing the benchmark case.

  Returns:
  - A BenchmarkInfo object.
  """
  # Extract the model name from the directory path. This uses the relative
  # path under the root model directory. If there are multiple segments,
  # additional ones will be placed in parentheses.
  model_name = os.path.relpath(benchmark_case_dir, benchmark_category_dir)
  # Now we have <model-name>/.../<iree-driver>__<target-arch>__<bench_mode>,
  # Remove the last segment.
  model_name = os.path.dirname(model_name)
  main, rest = os.path.split(model_name)
  if main:
    # Tags coming from directory structure.
    model_name = main
    model_tags = [re.sub(r"\W+", "-", rest)]
  else:
    # Tags coming from the name itself.
    model_name, rest = rest.split("-", 1)
    model_tags = rest.split(",")

  # Extract benchmark info from the directory path following convention:
  #   <iree-driver>__<target-architecture>__<benchmark_mode>
  root_immediate_dir = os.path.basename(benchmark_case_dir)
  iree_driver, target_arch, bench_mode = root_immediate_dir.split("__")

  model_source = os.path.basename(benchmark_category_dir)

  return BenchmarkInfo(model_name=model_name,
                       model_tags=model_tags,
                       model_source=model_source,
                       bench_mode=bench_mode.split(","),
                       runner=iree_driver,
                       device_info=device_info)


def filter_benchmarks_for_category(benchmark_category_dir: str,
                                   cpu_target_arch: str,
                                   gpu_target_arch: str,
                                   driver_filter: Optional[str],
                                   verbose: bool = False) -> Sequence[str]:
  """Filters benchmarks in a specific category for the given device.

  Args:
  - benchmark_category_dir: the directory to a specific benchmark category.
  - cpu_target_arch: CPU target architecture.
  - gpu_target_arch: GPU target architecture.
  - driver_filter: only run benchmarks for the given driver if not None.

  Returns:
  - A list containing all matched benchmark cases' directories.
  """
  matched_benchmarks = []

  # Go over all benchmarks in the model directory to find those matching the
  # current Android device's CPU/GPU architecture.
  for root, dirs, _ in os.walk(benchmark_category_dir):
    # Take the immediate directory name and try to see if it contains compiled
    # models and flagfiles. This relies on the following directory naming
    # convention:
    #   <iree-driver>__<target-architecture>__<benchmark_mode>
    root_immediate_dir = os.path.basename(root)
    segments = root_immediate_dir.split("__")
    if len(segments) != 3 or not segments[0].startswith("iree-"):
      continue

    iree_driver, target_arch, bench_mode = segments
    iree_driver = iree_driver[len("iree-"):].lower()
    target_arch = target_arch.lower()

    # We can choose this benchmark if it matches the driver and CPU/GPU
    # architecture.
    matched_driver = (driver_filter is None or
                      iree_driver == driver_filter.lower())
    matched_arch = (target_arch == cpu_target_arch or
                    target_arch == gpu_target_arch)
    should_choose = matched_driver and matched_arch
    if should_choose:
      matched_benchmarks.append(root)

    if verbose:
      print(f"dir: {root}")
      print(f"  iree_driver: {iree_driver}")
      print(f"  target_arch: {target_arch}")
      print(f"  bench_mode: {bench_mode}")
      print(f"  chosen: {should_choose}")

  return matched_benchmarks


def run_benchmarks_for_category(
    device_info: AndroidDeviceInfo,
    benchmark_category_dir: str,
    benchmark_case_dirs: Sequence[str],
    normal_benchmark_tool: str,
    traced_benchmark_tool: Optional[str],
    trace_capture_tool: Optional[str],
    verbose: bool = False
) -> Sequence[Tuple[BenchmarkInfo, Dict[str, Any], Dict[str, Any],
                    Optional[str]]]:
  """Runs all benchmarks on the Android device and reports results and captures.

  Args:
  - device_info: an AndroidDeviceInfo object.
  - benchmark_category_dir: the directory to a specific benchmark category.
  - benchmark_case_dirs: a list of benchmark case directories.
  - normal_benchmark_tool: the path to the normal benchmark tool.
  - traced_benchmark_tool: the path to the tracing-enabled benchmark tool.
  - trace_capture_tool: the path to the tool for collecting captured traces.

  Returns:
  - A list containing (BenchmarkInfo, context, results, capture-filename) tuples.
  """
  # Push the benchmark vmfb and tool files to the Android device first.
  adb_push_to_tmp_dir(os.path.join(benchmark_category_dir, VMFB_REL_PATH),
                      relative_dir=os.path.basename(benchmark_category_dir),
                      verbose=verbose)
  normal_benchmark_tool_path = adb_push_to_tmp_dir(normal_benchmark_tool,
                                                   relative_dir="normal-tools",
                                                   verbose=verbose)
  if traced_benchmark_tool is not None:
    traced_benchmark_tool_path = adb_push_to_tmp_dir(
        traced_benchmark_tool, relative_dir="traced-tools", verbose=verbose)

  results = []

  # Push all model artifacts to the device and run them.
  root_benchmark_dir = os.path.dirname(benchmark_category_dir)

  for benchmark_case_dir in benchmark_case_dirs:
    benchmark_info = compose_benchmark_info_object(device_info,
                                                   benchmark_category_dir,
                                                   benchmark_case_dir)
    print(f"--> benchmark: {benchmark_info} <--")

    android_relative_dir = os.path.relpath(benchmark_case_dir,
                                           root_benchmark_dir)
    adb_push_to_tmp_dir(os.path.join(benchmark_case_dir, MODEL_FLAGFILE_NAME),
                        android_relative_dir,
                        verbose=verbose)

    repetitions = get_benchmark_repetition_count(benchmark_info.runner)
    cmd = [
        "taskset",
        benchmark_info.deduce_taskset(),
        normal_benchmark_tool_path,
        f"--flagfile={MODEL_FLAGFILE_NAME}",
        f"--benchmark_repetitions={repetitions}",
        "--benchmark_format=json",
    ]

    resultjson = ""
    for i in range(3):
      try:
        resultjson = adb_execute_in_dir(cmd, android_relative_dir, verbose=verbose)
        break
      except subprocess.CalledProcessError:
        print(f"{cmd} failed. Retrying", file=sys.stderr)
        pass
    if resultjson == "":
      print("Benchmark failed. Skipping", file=sys.stderr)
      continue

    print(resultjson)
    resultjson = json.loads(resultjson)

    for previous_result in results:
      if previous_result[0] == benchmark_info:
        raise ValueError(f"Duplicated benchmark: {benchmark_info}")

    # If we have a tracing-enabled benchmark tool and the capture collecting
    # tool, catpure a trace of the benchmark run.
    capture_filename = None
    if traced_benchmark_tool is not None and trace_capture_tool is not None:
      run_cmd = [
          "TRACY_NO_EXIT=1", "taskset",
          benchmark_info.deduce_taskset(), traced_benchmark_tool_path,
          f"--flagfile={MODEL_FLAGFILE_NAME}"
      ]

      # Just launch the traced benchmark tool with TRACY_NO_EXIT=1 without
      # waiting for the adb command to complete as that won't happen.
      process = adb_start_in_dir(run_cmd, android_relative_dir, verbose=verbose)
      # But we do need to wait for its start; otherwise will see connection
      # failure when opening the catpure tool. Here we cannot just sleep a
      # certain amount of seconds---Pixel 4 seems to have an issue that will
      # make the trace collection step next stuck. Instead wait for the
      # benchmark result to be available.
      while True:
        line = process.stdout.readline()  # pytype: disable=attribute-error
        if line == "" and process.poll() is not None:  # Process completed
          raise ValueError("Cannot find benchmark result line in the log!")
        if verbose:
          print(line.strip())
        if re.match(r"^BM_.+/real_time", line) is not None:  # Result available
          break

      # Now it's okay to collect the trace via the capture tool. This will send
      # the signal to let the previously waiting benchmark tool to complete.
      capture_filename = re.sub(r" +", "-", str(benchmark_info)) + ".tracy"
      capture_cmd = [trace_capture_tool, "-f", "-o", capture_filename]
      capture_log = execute_cmd_and_get_output(capture_cmd, verbose=verbose)
      if verbose:
        print(capture_log)

      time.sleep(1)  # Some grace time.

    results.append((benchmark_info, resultjson["context"],
                    resultjson["benchmarks"], capture_filename))

  return results


def filter_and_run_benchmarks(
    device_info: AndroidDeviceInfo,
    root_build_dir: str,
    driver_filter: Optional[str],
    normal_benchmark_tool: str,
    traced_benchmark_tool: Optional[str],
    trace_capture_tool: Optional[str],
    verbose: bool = False) -> Tuple[BenchmarkResults, Sequence[str]]:
  """Filters and runs benchmarks in all categories for the given device.

  Args:
  - device_info: an AndroidDeviceInfo object.
  - root_build_dir: the root build directory.
  - driver_filter: only run benchmarks for the given driver if not None.
  - normal_benchmark_tool: the path to the normal benchmark tool.
  - traced_benchmark_tool: the path to the tracing-enabled benchmark tool.
  - trace_capture_tool: the path to the tool for collecting captured traces.
  """
  cpu_target_arch = CPU_ABI_TO_TARGET_ARCH_MAP[device_info.cpu_abi.lower()]
  gpu_target_arch = GPU_NAME_TO_TARGET_ARCH_MAP[device_info.gpu_name.lower()]

  root_benchmark_dir = os.path.join(root_build_dir, BENCHMARK_SUITE_REL_PATH)

  results = BenchmarkResults()
  captures = []

  for directory in os.listdir(root_benchmark_dir):
    benchmark_category_dir = os.path.join(root_benchmark_dir, directory)
    matched_benchmarks = filter_benchmarks_for_category(
        benchmark_category_dir=benchmark_category_dir,
        cpu_target_arch=cpu_target_arch,
        gpu_target_arch=gpu_target_arch,
        driver_filter=driver_filter,
        verbose=verbose)
    run_results = run_benchmarks_for_category(
        device_info=device_info,
        benchmark_category_dir=benchmark_category_dir,
        benchmark_case_dirs=matched_benchmarks,
        normal_benchmark_tool=normal_benchmark_tool,
        traced_benchmark_tool=traced_benchmark_tool,
        trace_capture_tool=trace_capture_tool,
        verbose=verbose)
    for info, context, runs, capture_filename in run_results:
      results.append_one_benchmark(info, context, runs)
      if capture_filename is not None:
        captures.append(capture_filename)

  # Attach commit information.
  results.set_commit(get_git_commit_hash("HEAD"))

  return (results, captures)


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
  parser.add_argument("--normal_benchmark_tool",
                      type=check_exe_path,
                      required=True,
                      help="Path to the normal iree-benchmark-module tool")
  parser.add_argument(
      "--traced_benchmark_tool",
      type=check_exe_path,
      default=None,
      help="Path to the tracing-enabled iree-benchmark-module tool")
  parser.add_argument("--trace_capture_tool",
                      type=check_exe_path,
                      default=None,
                      help="Path to the tool for collecting captured traces")
  parser.add_argument(
      "--driver",
      type=str,
      default=None,
      help="Only run benchmarks for a specific driver, e.g., 'vulkan'")
  parser.add_argument("-o",
                      dest="output",
                      default=None,
                      help="Path to the ouput file")
  parser.add_argument("--capture_tarball",
                      default=None,
                      help="Path to the tarball for captures")
  parser.add_argument("--no-clean",
                      action="store_true",
                      help="Do not clean up the temporary directory used for "
                      "benchmarking on the Android device")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")

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

  # Clear the benchmark directory on the Android device first just in case
  # there are leftovers from manual or failed runs.
  execute_cmd_and_get_output(["adb", "shell", "rm", "-rf", ANDROID_TMP_DIR],
                             verbose=args.verbose)

  # Tracy client and server communicate over port 8086 by default. If we want
  # to capture traces along the way, forward port via adb.
  if (args.traced_benchmark_tool is not None) and \
          (args.trace_capture_tool is not None):
    execute_cmd_and_get_output(["adb", "forward", "tcp:8086", "tcp:8086"])

    args.traced_benchmark_tool = os.path.realpath(args.traced_benchmark_tool)
    args.trace_capture_tool = os.path.realpath(args.trace_capture_tool)

  results, captures = filter_and_run_benchmarks(
      device_info=device_info,
      root_build_dir=args.build_dir,
      driver_filter=args.driver,
      normal_benchmark_tool=os.path.realpath(args.normal_benchmark_tool),
      traced_benchmark_tool=args.traced_benchmark_tool,
      trace_capture_tool=args.trace_capture_tool,
      verbose=args.verbose)

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
    for capture_filename in captures:
      os.remove(capture_filename)

    # Disable port forwarding.
    execute_cmd_and_get_output(["adb", "forward", "--remove", "tcp:8086"])

  if not args.no_clean:
    # Clear the benchmark directory on the Android device.
    execute_cmd_and_get_output(["adb", "shell", "rm", "-rf", ANDROID_TMP_DIR],
                               verbose=args.verbose)


if __name__ == "__main__":
  main(parse_arguments())
