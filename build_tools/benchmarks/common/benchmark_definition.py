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
import subprocess

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Sequence

# A map from CPU ABI to IREE's benchmark target architecture.
CPU_ABI_TO_TARGET_ARCH_MAP = {
    "arm64-v8a": "cpu-arm64-v8a",
}

# A map from GPU name to IREE's benchmark target architecture.
GPU_NAME_TO_TARGET_ARCH_MAP = {
    "adreno-640": "gpu-adreno",
    "adreno-650": "gpu-adreno",
    "adreno-660": "gpu-adreno",
    "adreno-730": "gpu-adreno",
    "mali-g77": "gpu-mali-valhall",
    "mali-g78": "gpu-mali-valhall",
}


@dataclass
class DriverInfo:
  """An object describing a IREE HAL driver.

  It includes the following characteristics:
  - pretty_name: the pretty name, e.g., 'IREE-DyLib'
  - device_type: the targeted device type, e.g., 'CPU'
  """

  pretty_name: str
  device_type: str


# A map for IREE driver names. This allows us to normalize driver names like
# mapping to more friendly ones and detach to keep driver names used in
# benchmark presentation stable.
IREE_DRIVERS_INFOS = {
    "iree-dylib": DriverInfo("IREE-Dylib", "CPU"),
    "iree-dylib-sync": DriverInfo("IREE-Dylib-Sync", "CPU"),
    "iree-vmvx": DriverInfo("IREE-VMVX", "CPU"),
    "iree-vmvx-sync": DriverInfo("IREE-VMVX-Sync", "CPU"),
    "iree-vulkan": DriverInfo("IREE-Vulkan", "GPU"),
}

IREE_PRETTY_NAMES_TO_DRIVERS = {
    v.pretty_name: k for k, v in IREE_DRIVERS_INFOS.items()
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
  cmd = " ".join(args)
  if verbose:
    print(f"cmd: {cmd}")
  try:
    return subprocess.run(args, check=True, text=True, **kwargs)
  except subprocess.CalledProcessError as exc:
    print((f"\n\nThe following command failed:\n\n{cmd}"
           f"\n\nReturn code: {exc.returncode}\n\n"))
    if exc.stdout:
      print(f"Stdout:\n\n{exc.stdout}\n\n")
    if exc.stderr:
      print(f"Stderr:\n\n{exc.stderr}\n\n")
    raise exc


def execute_cmd_and_get_output(args: Sequence[str],
                               verbose: bool = False,
                               **kwargs) -> str:
  """Executes a command and returns its stdout.

  Same as execute_cmd except captures stdout (and not stderr).
  """
  return execute_cmd(args, verbose=verbose, stdout=subprocess.PIPE,
                     **kwargs).stdout.strip()


class PlatformType(Enum):
  ANDROID = "Android"
  LINUX = "Linux"


@dataclass(frozen=True)
class DeviceInfo:
  """An object describing a device.

  It includes the following characteristics:
  - platform_type: the OS platform, e.g., 'Android'
  - model: the product model, e.g., 'Pixel-4'
  - cpu_abi: the CPU ABI, e.g., 'arm64-v8a'
  - cpu_features: the detailed CPU features, e.g., ['fphp', 'sve']
  - gpu_name: the GPU name, e.g., 'Mali-G77'
  """

  platform_type: PlatformType
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
    return f"{self.platform_type.value} device <{params}>"

  def get_iree_cpu_arch_name(self) -> str:
    arch = CPU_ABI_TO_TARGET_ARCH_MAP.get(self.cpu_abi.lower())
    if not arch:
      raise ValueError(f"Unrecognized CPU ABI: '{self.cpu_abi}'; "
                       "need to update the map")
    return arch

  def get_iree_gpu_arch_name(self) -> str:
    arch = GPU_NAME_TO_TARGET_ARCH_MAP.get(self.gpu_name.lower())
    if not arch:
      raise ValueError(f"Unrecognized GPU name: '{self.gpu_name}'; "
                       "need to update the map")
    return arch

  def get_cpu_arch_revision(self) -> str:
    if self.cpu_abi == "arm64-v8a":
      return self.__get_arm_cpu_arch_revision()
    raise ValueError("Unrecognized CPU ABI; need to update the list")

  def to_json_object(self) -> Dict[str, Any]:
    return {
        "platform_type": self.platform_type.value,
        "model": self.model,
        "cpu_abi": self.cpu_abi,
        "cpu_features": self.cpu_features,
        "gpu_name": self.gpu_name,
    }

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return DeviceInfo(PlatformType(json_object["platform_type"]),
                      json_object["model"], json_object["cpu_abi"],
                      json_object["cpu_features"], json_object["gpu_name"])

  def __get_arm_cpu_arch_revision(self) -> str:
    """Returns the ARM architecture revision."""

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


@dataclass(frozen=True)
class BenchmarkInfo:
  """An object describing the current benchmark.

  It includes the following benchmark characteristics:
  - model_name: the model name, e.g., 'MobileNetV2'
  - model_tags: a list of tags used to describe additional model information,
      e.g., ['imagenet']
  - model_source: the source of the model, e.g., 'TensorFlow'
  - bench_mode: a list of tags for benchmark mode,
      e.g., ['1-thread', 'big-core', 'full-inference']
  - runner: which runner is used for benchmarking, e.g., 'iree_vulkan', 'tflite'
  - device_info: an DeviceInfo object describing the device where benchmarks run
  """

  model_name: str
  model_tags: Sequence[str]
  model_source: str
  bench_mode: Sequence[str]
  runner: str
  device_info: DeviceInfo

  def __str__(self):
    # Get the target architecture and better driver name depending on the runner.
    driver_info = IREE_DRIVERS_INFOS.get(self.runner)
    if not driver_info:
      raise ValueError(
          f"Unrecognized runner '{self.runner}'; need to update the list")

    target_arch = None
    if driver_info.device_type == 'GPU':
      target_arch = "GPU-" + self.device_info.gpu_name
    elif driver_info.device_type == 'CPU':
      target_arch = "CPU-" + self.device_info.get_cpu_arch_revision()
    else:
      raise ValueError(
          f"Unrecognized device type '{driver_info.device_type}' of the runner '{self.runner}'"
      )

    if self.model_tags:
      tags = ",".join(self.model_tags)
      model_part = f"{self.model_name} [{tags}] ({self.model_source})"
    else:
      model_part = f"{self.model_name} ({self.model_source})"
    device_part = f"{self.device_info.model} ({target_arch})"
    mode = ",".join(self.bench_mode)

    return f"{model_part} {mode} with {driver_info.pretty_name} @ {device_part}"

  @staticmethod
  def from_device_info_and_name(device_info: DeviceInfo, name: str):
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
    return BenchmarkInfo(model_name, model_tags, model_source, bench_mode,
                         runner, device_info)

  def to_json_object(self) -> Dict[str, Any]:
    return {
        "model_name": self.model_name,
        "model_tags": self.model_tags,
        "model_source": self.model_source,
        "bench_mode": self.bench_mode,
        "runner": self.runner,
        "device_info": self.device_info.to_json_object(),
    }

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return BenchmarkInfo(model_name=json_object["model_name"],
                         model_tags=json_object["model_tags"],
                         model_source=json_object["model_source"],
                         bench_mode=json_object["bench_mode"],
                         runner=json_object["runner"],
                         device_info=DeviceInfo.from_json_object(
                             json_object["device_info"]))


@dataclass
class BenchmarkRun(object):
  """An object describing a single run of the benchmark binary.

  - benchmark_info: a BenchmarkInfo object describing the benchmark setup.
  - context: the benchmark context returned by the benchmarking framework.
  - results: the benchmark results returned by the benchmarking framework.
  """
  benchmark_info: BenchmarkInfo
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
        BenchmarkInfo.from_json_object(json_object["benchmark_info"]),
        json_object["context"], json_object["results"])


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
