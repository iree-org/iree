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
"""Utilities for describing Android benchmarks."""

import json
import re
import subprocess

__all__ = ["AndroidDeviceInfo", "BenchmarkInfo", "BenchmarkResults"]

# A map for IREE driver names. This allows us to normalize driver names like
# mapping to more friendly ones and detach to keep driver names used in
# benchmark presentation stable.
IREE_DRIVER_NAME_MAP = {
    "iree_llvmaot": "IREE-Dylib",
    "iree_vmvx": "IREE-VMVX",
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
  """An object describing the current Android Device.

  It includes the following phone characteristics:
  - model: the product model, e.g., 'Pixel-4'
  - cpu_abi: the CPU ABI, e.g., 'arm64-v8a'
  - cpu_features: the detailed CPU features, e.g., ['fphp', 'sve']
  - gpu_name: the GPU name, e.g., 'Mali-G77'
  """

  def __init__(self, model, cpu_abi, cpu_features, gpu_name, verbose=False):
    self.model = get_android_device_model(verbose)
    self.cpu_abi = get_android_cpu_abi(verbose)
    self.cpu_features = get_android_cpu_features(verbose)
    self.gpu_name = get_android_gpu_name(verbose)

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

  def to_json_object(self):
    return {
        "model": self.model,
        "cpu_abi": self.cpu_abi,
        "cpu_features": self.cpu_features,
        "gpu_name": self.gpu_name,
    }

  @staticmethod
  def from_json_object(json_object):
    return AndroidDeviceInfo(json_object["model"], json_object["cpu_abi"],
                             json_object["cpu_features"],
                             json_object["gpu_name"])

  @staticmethod
  def from_adb(verbose=False):
    return AndroidDeviceInfo(get_android_device_model(verbose),
                             get_android_cpu_abi(verbose),
                             get_android_gpu_name(verbose),
                             get_android_cpu_features(verbose))


class BenchmarkInfo(object):
  """An object describing the current benchmark.

  It includes the following benchmark characteristics:
  - model_name: the model name, e.g., 'MobileNetV2'
  - model_tags: a list of tags used to describe additional model information,
      e.g., ['imagenet']
  - model_source: the source of the model, e.g., 'TensorFlow'
  - runner: which runner is used for benchmarking, e.g., 'iree_vulkan', 'tflite'
  - device_info: an AndroidDeviceInfo object describing the phone where
      bnechmarks run
  """

  def __init__(self, model_name, model_tags, model_source, runner, device_info):
    self.model_name = model_name
    self.model_tags = model_tags
    self.model_source = model_source
    self.runner = runner
    self.device_info = device_info

  def __str__(self):
    # Get the target architecture and better driver name depending on the runner.
    target_arch = ""
    driver = ""
    if self.runner == "iree_vulkan":
      target_arch = "GPU-" + self.device_info.gpu_name
      driver = IREE_DRIVER_NAME_MAP[self.runner]
    elif self.runner == "iree_llvmaot" or self.runner == "iree_vmvx":
      target_arch = "CPU-" + self.device_info.get_arm_arch_revision()
      driver = IREE_DRIVER_NAME_MAP[self.runner]
    else:
      raise ValueError("Unrecognized runner; need to update the list")

    if self.model_tags:
      tags = ",".join(self.model_tags)
      model_part = f"{self.model_name} ({tags}) [{self.model_source}]"
    else:
      model_part = f"{self.model_name} [{self.model_source}]"
    phone_part = f"{self.device_info.model} ({target_arch})"

    return f"{model_part} with {driver} @ {phone_part}"

  def to_json_object(self):
    return {
        "model_name": self.model_name,
        "model_tags": self.model_tags,
        "model_source": self.model_source,
        "runner": self.runner,
        "device_info": self.device_info.to_json_object(),
    }

  @staticmethod
  def from_json_object(json_object):
    return BenchmarkInfo(
        json_object["model_name"], json_object["model_tags"],
        json_object["model_source"], json_object["runner"],
        AndroidDeviceInfo.from_json_object(json_object["device_info"]))


class BenchmarkResults(object):
  """An object describing a set of benchmarks for one particular commit.

    It contains the following fields:
    - commit: the commit SHA for this set of benchmarks.
    - benchmarks: a list of benchmarks, each with
      - benchmark: a BenchmarkInfo object
      - context: the context for running the benchmarks
      - results: results for all benchmark runs
    """

  def __init__(self):
    self.commit = "<unknown>"
    self.benchmarks = []

  def set_commit(self, commit):
    self.commit = commit

  def append_one_benchmark(self, benchmark_info, run_context, run_results):
    """Appends the results for one benchmark."""
    self.benchmarks.append({
        "benchmark": benchmark_info,
        "context": run_context,
        "results": run_results,
    })

  def to_json_str(self):
    json_object = {"commit": self.commit, "benchmarks": []}
    for benchmark in self.benchmarks:
      json_object["benchmarks"].append({
          "benchmark": benchmark["benchmark"].to_json_object(),
          "context": benchmark["context"],
          "results": benchmark["results"],
      })
    return json.dumps(json_object)

  @staticmethod
  def from_json_str(json_str):
    json_object = json.loads(json_str)
    results = BenchmarkResults()
    results.set_commit(json_object["commit"])
    for benchmark in json_object["benchmarks"]:
      results.append_one_benchmark(
          BenchmarkInfo.from_json_object(benchmark["benchmark"]),
          benchmark["context"], benchmark["results"])
    return results
