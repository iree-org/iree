# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for describing Android benchmarks.

This file provides common and structured representation of Android devices,
benchmark definitions, and benchmark result collections, so that they can be
shared between different stages of the same benchmark pipeline.
"""

import json
import re
import subprocess

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

# A map for IREE driver names. This allows us to normalize driver names like
# mapping to more friendly ones and detach to keep driver names used in
# benchmark presentation stable.
IREE_DRIVERS_TO_PRETTY_NAMES = {
    "iree-dylib": "IREE-Dylib",
    "iree-dylib-sync": "IREE-Dylib-Sync",
    "iree-vmvx": "IREE-VMVX",
    "iree-vmvx-sync": "IREE-VMVX-Sync",
    "iree-vulkan": "IREE-Vulkan",
}

IREE_PRETTY_NAMES_TO_DRIVERS = {
    v: k for k, v in IREE_DRIVERS_TO_PRETTY_NAMES.items()
}


def execute_cmd(args: Sequence[str],
                verbose: bool = False,
                **kwargs) -> subprocess.CompletedProcess:
  """Executes a command and returns the completed process.

  A thin wrapper around subprocess.run that sets some useful defaults and
  optionally prints out the command being run.

  Raises:
    CalledProcessError if the command fails.
  """
  if verbose:
    cmd = " ".join(args)
    print(f"cmd: {cmd}")
  return subprocess.run(args, check=True, text=True, **kwargs)


def execute_cmd_and_get_output(args: Sequence[str],
                               verbose: bool = False,
                               **kwargs) -> str:
  """Executes a command and returns its stdout.

  Same as execute_cmd except captures stdout (and not stderr).
  """
  return execute_cmd(args, verbose=verbose, stdout=subprocess.PIPE,
                     **kwargs).stdout.strip()


def get_android_device_model(verbose: bool = False) -> str:
  """Returns the Android device model."""
  model = execute_cmd_and_get_output(
      ["adb", "shell", "getprop", "ro.product.model"], verbose=verbose)
  model = re.sub(r"\W+", "-", model)
  return model


def get_android_cpu_abi(verbose: bool = False) -> str:
  """Returns the CPU ABI for the Android device."""
  return execute_cmd_and_get_output(
      ["adb", "shell", "getprop", "ro.product.cpu.abi"], verbose=verbose)


def get_android_cpu_features(verbose: bool = False) -> Sequence[str]:
  """Returns the CPU features for the Android device."""
  cpuinfo = execute_cmd_and_get_output(["adb", "shell", "cat", "/proc/cpuinfo"],
                                       verbose=verbose)
  features = []
  for line in cpuinfo.splitlines():
    if line.startswith("Features"):
      _, features = line.split(":")
      return features.strip().split()
  return features


def get_android_gpu_name(verbose: bool = False) -> str:
  """Returns the GPU name for the Android device."""
  vkjson = execute_cmd_and_get_output(["adb", "shell", "cmd", "gpu", "vkjson"],
                                      verbose=verbose)
  vkjson = json.loads(vkjson)
  name = vkjson["devices"][0]["properties"]["deviceName"]

  # Perform some canonicalization:

  # - Adreno GPUs have raw names like "Adreno (TM) 650".
  name = name.replace("(TM)", "")

  # Replace all consecutive non-word characters with a single hypen.
  name = re.sub(r"\W+", "-", name)

  return name


@dataclass
class AndroidDeviceInfo:
  """An object describing the current Android Device.

  It includes the following phone characteristics:
  - model: the product model, e.g., 'Pixel-4'
  - cpu_abi: the CPU ABI, e.g., 'arm64-v8a'
  - cpu_features: the detailed CPU features, e.g., ['fphp', 'sve']
  - gpu_name: the GPU name, e.g., 'Mali-G77'
  """

  model: str
  cpu_abi: str
  cpu_features: Sequence[str]
  gpu_name: str

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

  def get_arm_arch_revision(self) -> str:
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

  def to_json_object(self) -> Dict[str, Any]:
    return {
        "model": self.model,
        "cpu_abi": self.cpu_abi,
        "cpu_features": self.cpu_features,
        "gpu_name": self.gpu_name,
    }

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return AndroidDeviceInfo(json_object["model"], json_object["cpu_abi"],
                             json_object["cpu_features"],
                             json_object["gpu_name"])

  @staticmethod
  def from_adb(verbose: bool = False):
    return AndroidDeviceInfo(get_android_device_model(verbose),
                             get_android_cpu_abi(verbose),
                             get_android_cpu_features(verbose),
                             get_android_gpu_name(verbose))


@dataclass
class BenchmarkOrStatisticInfo:
  """An object describing the current benchmark or statistic.

  It includes the following characteristics:
  - model_name: the model name, e.g., 'MobileNetV2'
  - model_tags: a list of tags used to describe additional model information,
      e.g., ['imagenet']
  - model_source: the source of the model, e.g., 'TensorFlow'
  - bench_mode: a list of tags for benchmark mode,
      e.g., ['1-thread', 'big-core', 'full-inference']
  - runner: which runner is used for benchmarking, e.g., 'iree_vulkan', 'tflite'
      optional for statistics
  - device_info: an AndroidDeviceInfo object describing the phone where
      benchmarks run; optional for statistics
  - statistic: breadcrumb hierarchical description of the statistic;
      missing for benchmarks
  """

  # Common fields for benchmarks and statistics
  model_name: str
  model_tags: Sequence[str]
  model_source: str
  bench_mode: Sequence[str]

  # Fields that might be missing for statistics
  runner: Optional[str] = None
  device_info: Optional[AndroidDeviceInfo] = None

  # Fields missing for benchmarks
  statistic: Sequence[str] = ()

  def __str__(self):
    # Get the target architecture and pretty driver name depending on the runner.
    target_arch = ""
    driver = ""
    if self.runner is None:
      pass
    elif self.runner == "iree-vulkan":
      if self.device_info:
        target_arch = "GPU-" + self.device_info.gpu_name
      driver = IREE_DRIVERS_TO_PRETTY_NAMES[self.runner]
    elif (self.runner == "iree-dylib" or self.runner == "iree-dylib-sync" or
          self.runner == "iree-vmvx" or self.runner == "iree-vmvx-sync"):
      if self.device_info:
        target_arch = "CPU-" + self.device_info.get_arm_arch_revision()
      driver = IREE_DRIVERS_TO_PRETTY_NAMES[self.runner]
    else:
      raise ValueError(
          f"Unrecognized runner '{self.runner}'; need to update the list")

    if self.model_tags:
      tags = ",".join(self.model_tags)
      model_part = f"{self.model_name} [{tags}] ({self.model_source})"
    else:
      model_part = f"{self.model_name} ({self.model_source})"
    mode = ",".join(self.bench_mode)

    stat = " statistic:" + "-".join(self.statistic) if self.statistic else ""

    full_str = f"{model_part} {mode}{stat}"
    if self.runner:
      full_str += f" with {driver}"

    if self.device_info:
      phone_part = f"{self.device_info.model} ({target_arch})"
      full_str += f" @ {phone_part}"
    return full_str

  @staticmethod
  def from_device_info_and_name(device_info: AndroidDeviceInfo, name: str):
    (
        model_name,
        model_tags,
        model_source,
        bench_mode,
        _,  # "with"
        runner,
        _,  # "@"
        model,
        _,  # Device Info
    ) = name.split()
    model_source = model_source.strip("()")
    model_tags = model_tags.strip("[]").split(",")
    bench_mode = bench_mode.split(",")
    runner = IREE_PRETTY_NAMES_TO_DRIVERS.get(runner)
    return BenchmarkOrStatisticInfo(model_name, model_tags, model_source,
                                    bench_mode, runner, device_info)

  def deduce_taskset(self) -> str:
    """Deduces the CPU affinity taskset mask according to benchmark modes."""
    # TODO: we actually should check the number of cores the phone have.
    if "big-core" in self.bench_mode:
      return "80" if "1-thread" in self.bench_mode else "f0"
    if "little-core" in self.bench_mode:
      return "08" if "1-thread" in self.bench_mode else "0f"

    # Not specified: use the 7th core.
    return "80"

  def to_json_object(self) -> Dict[str, Any]:
    json_object = {
        "model_name": self.model_name,
        "model_tags": self.model_tags,
        "model_source": self.model_source,
        "bench_mode": self.bench_mode,
    }
    if self.runner:
      json_object["runner"] = self.runner
    if self.device_info:
      json_object["device_info"] = self.device_info.to_json_object()
    if self.statistic:
      json_object["statistic"] = self.statistic
    return json_object

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    device_info = None
    if "device_info" in json_object:
      device_info = AndroidDeviceInfo.from_json_object(
          json_object["device_info"])

    return BenchmarkOrStatisticInfo(model_name=json_object["model_name"],
                                    model_tags=json_object["model_tags"],
                                    model_source=json_object["model_source"],
                                    bench_mode=json_object["bench_mode"],
                                    runner=json_object.get("runner"),
                                    device_info=device_info,
                                    statistic=json_object.get("statistic"))


@dataclass
class BenchmarkRun(object):
  """An object describing a single run of the benchmark binary.

  - benchmark_info: a BenchmarkOrStatisticInfo object describing the benchmark setup.
  - context: the benchmark context returned by the benchmarking framework.
  - results: the benchmark results returned by the benchmarking framework.
  """
  benchmark_info: BenchmarkOrStatisticInfo
  context: Dict[str, Any]
  results: Sequence[Dict[str, Any]]

  def to_json_object(self) -> Dict[str, Any]:
    return {
        "benchmark_info": self.benchmark_info.to_json_object(),
        "context": self.context,
        "results": self.results,
    }

  def to_json_str(self) -> str:
    return json.dumps(self.to_json_object())

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return BenchmarkRun(
        BenchmarkOrStatisticInfo.from_json_object(
            json_object["benchmark_info"]), json_object["context"],
        json_object["results"])


class BenchmarkResults(object):
  """An object describing a set of benchmarks for one particular commit.

    It contains the following fields:
    - commit: the commit SHA for this set of benchmarks.
    - benchmarks: a list of BenchmarkRun objects
    """

  def __init__(self):
    self.commit = "<unknown>"
    self.benchmarks = []

  def set_commit(self, commit: str):
    self.commit = commit

  def merge(self, other):
    if self.commit != other.commit:
      raise ValueError("Inconsistent pull request commit")
    self.benchmarks.extend(other.benchmarks)

  def get_aggregate_time(self, benchmark_index: int, kind: str) -> int:
    """Returns the Google Benchmark aggreate time for the given kind.

      Args:
      - benchmark_index: the benchmark's index.
      - kind: what kind of aggregate time to get; choices:
        'mean', 'median', 'stddev'.
      """
    time = None
    for bench_case in self.benchmarks[benchmark_index].results:
      if bench_case["name"].endswith(f"real_time_{kind}"):
        if bench_case["time_unit"] != "ms":
          raise ValueError(f"Expected ms as time unit")
        time = int(round(bench_case["real_time"]))
        break
    if time is None:
      raise ValueError(f"Cannot found real_time_{kind} in benchmark results")
    return time

  def to_json_str(self) -> str:
    json_object = {"commit": self.commit, "benchmarks": []}
    json_object["benchmarks"] = [b.to_json_object() for b in self.benchmarks]
    return json.dumps(json_object)

  @staticmethod
  def from_json_str(json_str: str):
    json_object = json.loads(json_str)
    results = BenchmarkResults()
    results.set_commit(json_object["commit"])
    results.benchmarks = [
        BenchmarkRun.from_json_object(b) for b in json_object["benchmarks"]
    ]
    return results


@dataclass
class StatisticInstance(object):
  """An object describing a single instance of a statistic.

  - statistic_info: a BenchmarkOrStatisticInfo object describing the statistic setup.
  - value: the value for this statistic instance.
  """
  statistic_info: BenchmarkOrStatisticInfo
  value: Any

  def to_json_object(self) -> Dict[str, Any]:
    return {
        "statistic_info": self.statistic_info.to_json_object(),
        "value": self.value,
    }

  def to_json_str(self) -> str:
    return json.dumps(self.to_json_object())

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return StatisticInstance(
        BenchmarkOrStatisticInfo.from_json_object(
            json_object["statistic_info"]), json_object["value"])


class StatisticResults(object):
  """An object describing a set of statistics for one particular commit.

    It contains the following fields:
    - commit: the commit SHA for this set of benchmarks.
    - statistics: a list of statistics.
    """

  def __init__(self):
    self.commit = "<unknown>"
    self.statistics = []

  def set_commit(self, commit: str):
    self.commit = commit

  def merge(self, other):
    if self.commit != other.commit:
      raise ValueError("Inconsistent pull request commit")
    self.statistics.extend(other.statistics)

  def to_json_str(self) -> str:
    json_object = {"commit": self.commit}
    json_object["statistics"] = [s.to_json_object() for s in self.statistics]
    return json.dumps(json_object)

  @staticmethod
  def from_json_str(json_str: str):
    json_object = json.loads(json_str)
    results = BenchmarkResults()
    results.set_commit(json_object["commit"])
    results.statistics = [
        StatisticInstance.from_json_object(b) for b in json_object["statistics"]
    ]
    return results
