#!/usr/bin/env python3

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs all matched benchmark suites onto an Android device.

This script probes the Android phone via adb and uses the device information
to filter and run suitable benchmarks on it.

It expects that `adb` is installed, and there is an `iree-benchmark-module`
tool cross-compiled towards Android. It also expects the benchmark artifacts
are generated using `ninja iree-benchmark-suites`.

Example usages:

  python3 run_benchmarks.py \
    --benchmark_tool=/path/to/android/target/iree-benchmark_module \
    /path/to/host/build/dir
"""

import argparse
import json
import os
import re
import subprocess

# Relative path against build directory.
BENCHMARK_SUITE_REL_PATH = "benchmark_suites"
# Relative path against root benchmark suit directory.
PYTON_MODEL_REL_PATH = "tf_models"

# The artifact's filename for compiled Python models.
MODEL_FLAGFILE_NAME = "flagfile"
# The flagfile's filename for compiled Python models.
MODEL_VMFB_NAME = "compiled.vmfb"

# Root directory to perform benchmarks in on the Android device.
ANDROID_TMP_DIR = "/data/local/tmp/iree-benchmarks"

BENCHMARK_REPETITIONS = 10

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

# A map for IREE driver names. This allows us to normalize driver names like
# mapping to more friendly ones and detach to keep driver names used in
# benchmark presentation stable.
IREE_DRIVER_NAME_MAP = {
    "iree_llvmaot": "IREE-Dylib",
    "iree_vulkan": "IREE-Vulkan",
}


def execute(args,
            capture_output=False,
            treat_io_as_text=True,
            verbose=False,
            **kwargs):
  """Executes a command."""
  if verbose:
    cmd = " ".join(args)
    print(f"cmd: {cmd}")
  return subprocess.run(args,
                        check=True,
                        capture_output=capture_output,
                        text=treat_io_as_text,
                        **kwargs)


def get_git_commit_hash(commit):
   return execute(['git', 'rev-parse', commit],
                   cwd=os.path.dirname(os.path.realpath(__file__)),
                   capture_output=True).stdout.strip()


def get_android_device_model(verbose=False):
  """Returns the Android device model."""
  model = execute(["adb", "shell", "getprop", "ro.product.model"],
                  capture_output=True,
                  verbose=verbose).stdout.strip()
  model = re.sub(r"\W+", "-", model)
  return model


def get_android_cpu_abi(verbose=False):
  """Returns the CPU ABI for the Android device."""
  return execute(["adb", "shell", "getprop", "ro.product.cpu.abi"],
                 capture_output=True,
                 verbose=verbose).stdout.strip()


def get_android_cpu_features(verbose=False):
  """Returns the CPU features for the Android device."""
  cpuinfo = execute(["adb", "shell", "cat", "/proc/cpuinfo"],
                    capture_output=True,
                    verbose=verbose).stdout.strip()
  features = []
  for line in cpuinfo.splitlines():
    if line.startswith("Features"):
      _, features = line.split(":")
      return features.strip().split()
  return features


def get_android_gpu_name(verbose=False):
  """Returns the GPU name for the Android device."""
  vkjson = execute(["adb", "shell", "cmd", "gpu", "vkjson"],
                   capture_output=True,
                   verbose=verbose).stdout.strip()
  vkjson = json.loads(vkjson)
  name = vkjson["devices"][0]["properties"]["deviceName"]

  # Perform some canonicalization:

  # - Adreno GPUs have raw names like "Adreno (TM) 650".
  name = name.replace("(TM)", "")

  # Replace all consecutive non-word characters with a single hypen.
  name = re.sub(r"\W+", "-", name)

  return name


class AndroidDeviceInfo(object):

  def __init__(self, verbose=False):
    self.model = get_android_device_model(verbose)
    self.cpu_abi = get_android_cpu_abi(verbose)
    self.gpu_name = get_android_gpu_name(verbose)
    self.cpu_features = get_android_cpu_features(verbose)

  def __str__(self):
    features = ", ".join(self.cpu_features)
    params = [
        f"model='{self.model}'",
        f"cpu_abi='{self.cpu_abi}'",
        f"gpu_name='{self.gpu_name}'",
        f"cpu_features=[{features}]",
    ]
    params = ", ".join(params)
    return f"Android device <{params}>"

  def get_arm_arch_revision(self):
    """Returns the ARM architecture revision."""
    if self.cpu_abi != "arm64-v8a":
      raise ValueError("Unrecognized ARM CPU ABI; need to update the list")

    # CPU features for ARMv8 revisions.
    # From https://en.wikichip.org/wiki/arm/armv8#ARMv8_Extensions_and_Processor_Features
    rev1_features = ["atomics", "asimdrdm"]
    rev2_features = [
        "fphp", "dcpop", "sha3", "sm3", "sm4", "asimddp", "sha512", "sve"
    ]

    rev = "ARMv8-A"
    if any([f in self.cpu_features for f in rev1_features]):
      rev = "ARMv8.1-A"
    if any([f in self.cpu_features for f in rev2_features]):
      rev = "ARMv8.2-A"
    return rev


def adb_push_to_tmp_dir(content, relative_dir, verbose=False):
  """Pushes content onto the Android device.

  Args:
  - content: the full path to the source file.
  - relative_dir: the directory to push to; relative to ANDROID_TMP_DIR.

  Returns:
  - The full path to the content on the Android device.
  """
  filename = os.path.basename(content)
  android_path = f"{ANDROID_TMP_DIR}/{relative_dir}/{filename}"
  execute(["adb", "push", os.path.abspath(content), android_path],
          verbose=verbose)
  return android_path


def adb_execute_in_path(cmd_args, relative_dir, verbose=False):
  """Executes command with adb shell in a directory.

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

  return execute(cmd, capture_output=True, verbose=verbose).stdout


def compose_benchmark_name(device_info, root_build_dir, model_benchmark_dir):
  """Creates a friendly name to describe the benchmark.

    Args:
    - device_info: an AndroidDeviceInfo object.
    - root_build_dir: the root build directory.
    - model_benchmark_dirs: a directory containing model benchmarks.

    Returns:
    - A string of the format:
      "<model-name> [<source>] <benchmark-mode> with <iree-driver>"
      " @ <device-model> (<target-arch>)"
  """
  model_root_dir = os.path.join(root_build_dir, BENCHMARK_SUITE_REL_PATH,
                                PYTON_MODEL_REL_PATH)

  # Extract the model name from the directory path. This uses the relative
  # path under the root model directory. If there are multiple segments,
  # additional ones will be placed in parentheses.
  model_name = os.path.relpath(model_benchmark_dir, model_root_dir)
  model_name = os.path.dirname(model_name)  # Remove IREE driver segment
  main, rest = os.path.split(model_name)
  rest = re.sub(r"\W+", "-", rest)
  model_name = f"{main} ({rest})" if main else f"{rest}"

  # Extract benchmark info from the directory path following convention:
  #   <iree-driver>__<target-architecture>__<benchmark_mode>
  root_immediate_dir = os.path.basename(model_benchmark_dir)
  iree_driver, target_arch, bench_mode = root_immediate_dir.split("__")

  # Get the target architecture depending on the IREE driver.
  target_arch = ""
  driver = ""
  if iree_driver == "iree_vulkan":
    target_arch = "GPU-" + device_info.gpu_name
    driver = IREE_DRIVER_NAME_MAP[iree_driver]
  elif iree_driver == "iree_llvmaot":
    target_arch = "CPU-" + device_info.get_arm_arch_revision()
    driver = IREE_DRIVER_NAME_MAP[iree_driver]
  else:
    raise ValueError("Unrecognized IREE driver; need to update the list")

  model_driver = f"{model_name} [TensorFlow] {bench_mode} with {driver}"
  device_arch = f"{device_info.model} ({target_arch})"
  return model_driver + " @ " + device_arch


def filter_python_model_benchmark_suite(device_info,
                                        root_build_dir,
                                        verbose=False):
  """Filters Python model benchmark suite for the given CPU/GPU target.

  Args:
  - device_info: an AndroidDeviceInfo object.
  - root_build_dir: the root build directory.

  Returns:
  - A list containing all matched benchmark's directories.
  """
  cpu_target_arch = CPU_ABI_TO_TARGET_ARCH_MAP[device_info.cpu_abi.lower()]
  gpu_target_arch = GPU_NAME_TO_TARGET_ARCH_MAP[device_info.gpu_name.lower()]

  model_root_dir = os.path.join(root_build_dir, BENCHMARK_SUITE_REL_PATH,
                                PYTON_MODEL_REL_PATH)
  matched_benchmarks = []

  # Go over all benchmarks in the model directory to find those matching the
  # current Android device's CPU/GPU architecture.
  for root, dirs, files in os.walk(model_root_dir):
    # Take the immediate directory name and try to see if it contains compiled
    # models and flagfiles. This replies on the following directory naming
    # convention:
    #   <iree-driver>__<target-architecture>__<benchmark_mode>
    root_immediate_dir = os.path.basename(root)
    segments = root_immediate_dir.split("__")
    if len(segments) != 3 or not segments[0].startswith("iree_"):
      continue

    iree_driver, target_arch, bench_mode = segments
    target_arch = target_arch.lower()
    # We can choose this benchmark if it matches the CPU/GPU architecture.
    should_choose = (target_arch == cpu_target_arch or
                     target_arch == gpu_target_arch)
    if should_choose:
      matched_benchmarks.append(root)

    if verbose:
      print(f"dir: {root}")
      print(f"  iree_driver: {iree_driver}")
      print(f"  target_arch: {target_arch}")
      print(f"  bench_mode: {bench_mode}")
      print(f"  chosen: {should_choose}")

  return matched_benchmarks


def run_python_model_benchmark_suite(device_info,
                                     root_build_dir,
                                     model_benchmark_dirs,
                                     benchmark_tool,
                                     verbose=False):
  """Runs all model benchmarks on the Android device and report results.

  Args:
  - device_info: an AndroidDeviceInfo object.
  - root_build_dir: the root build directory.
  - model_benchmark_dirs: a list of model benchmark directories.
  - benchmark_tool: the path to the benchmark tool.

  Returns:
  - A dictionary containing maps from benchmark names to the time (ms).
  """
  # Push the benchmark tool to the Android device first.
  android_tool_path = adb_push_to_tmp_dir(benchmark_tool,
                                          relative_dir="tools",
                                          verbose=verbose)

  model_root_dir = os.path.join(root_build_dir, BENCHMARK_SUITE_REL_PATH,
                                PYTON_MODEL_REL_PATH)

  results = {}

  # Push all model artifacts to the device and run them.
  for model_benchmark_dir in model_benchmark_dirs:
    benchmark_name = compose_benchmark_name(device_info, root_build_dir,
                                            model_benchmark_dir)
    print(f"--> benchmark: {benchmark_name} <--")
    android_relative_dir = os.path.relpath(model_benchmark_dir, model_root_dir)
    adb_push_to_tmp_dir(os.path.join(model_benchmark_dir, MODEL_VMFB_NAME),
                        android_relative_dir,
                        verbose=verbose)
    android_flagfile_path = adb_push_to_tmp_dir(os.path.join(
        model_benchmark_dir, MODEL_FLAGFILE_NAME),
                                                android_relative_dir,
                                                verbose=verbose)

    cmd = [
        android_tool_path,
        f"--flagfile={android_flagfile_path}",
        f"--benchmark_repetitions={BENCHMARK_REPETITIONS}",
        "--benchmark_format=json",
    ]
    resultjson = adb_execute_in_path(cmd, android_relative_dir, verbose=verbose)

    print(resultjson)
    resultjson = json.loads(resultjson)

    if benchmark_name in results:
      raise ValueError(f"Duplicated benchmark name: {benchmark_name}")

    avg_time = None
    for benchmark in resultjson["benchmarks"]:
      if benchmark["name"].endswith("real_time_mean"):
        if benchmark["time_unit"] != "ms":
          raise ValueError(
              f"Expected ms as time unit but found {benchmark['time_unit']}")
        avg_time = int(benchmark["real_time"])
        break
    if avg_time is None:
      raise ValueError(f"Cannot find average time")

    results[benchmark_name] = avg_time

  return results


def parse_arguments():
  """Parses command-line options."""

  def check_dir_path(path):
    if os.path.isdir(path):
      return path
    else:
      raise NotADirectoryError(path)

  def check_exe_path(path):
    if os.access(path, os.X_OK):
      return path
    else:
      raise ValueError(f"'{path}' is not an executable")

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "build_dir",
      metavar="<build-dir>",
      type=check_dir_path,
      help="Path to the build directory containing benchmark suites")
  parser.add_argument("--benchmark_tool",
                      type=check_exe_path,
                      default=None,
                      help="Path to the iree-benchmark-module tool (default to "
                      "iree/tools/iree-benchmark-module under <build-dir>)")
  parser.add_argument("-o",
                      dest="output",
                      default=None,
                      help="Path to the ouput file")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")

  args = parser.parse_args()

  if args.benchmark_tool is None:
    args.benchmark_tool = os.path.join(args.build_dir, "iree", "tools",
                                       "iree-benchmark-module")

  return args


def main(args):
  device_info = AndroidDeviceInfo()
  print(device_info)

  if device_info.cpu_abi.lower() not in CPU_ABI_TO_TARGET_ARCH_MAP:
    raise ValueError(f"Unrecognized CPU ABI: '{device_info.cpu_abi}'; "
                     "need to update the map")
  if device_info.gpu_name.lower() not in GPU_NAME_TO_TARGET_ARCH_MAP:
    raise ValueError(f"Unrecognized GPU name: '{device_info.gpu_name}'; "
                     "need to update the map")

  benchmarks = filter_python_model_benchmark_suite(device_info, args.build_dir,
                                                   args.verbose)
  results = run_python_model_benchmark_suite(device_info,
                                             args.build_dir,
                                             benchmarks,
                                             args.benchmark_tool,
                                             verbose=args.verbose)

  # Attach commit information.
  head_commit = get_git_commit_hash("HEAD")
  results = {"commit": head_commit, "benchmarks": results}

  if args.output is not None:
    with open(args.output, "w") as f:
      json.dump(results, f)
  print(results)


if __name__ == "__main__":
  main(parse_arguments())
