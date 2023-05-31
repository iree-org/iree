# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for describing benchmarks.

This file provides common and structured representation of devices, benchmark
definitions, and benchmark result collections, so that they can be shared
between different stages of the same benchmark pipeline.
"""

import json
import pathlib
import re
import subprocess

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from e2e_test_framework.definitions import common_definitions

# A map from CPU ABI to IREE's legacy benchmark target architecture.
CPU_ABI_TO_LEGACY_TARGET_ARCH_MAP = {
    "arm64-v8a": "cpu-arm64-v8a",
    "x86_64-cascadeLake": "cpu-x86_64-cascadelake",
}

# A map from GPU name to IREE's legacy benchmark target architecture.
GPU_NAME_TO_LEGACY_TARGET_ARCH_MAP = {
    "adreno-640": "gpu-adreno",
    "adreno-650": "gpu-adreno",
    "adreno-660": "gpu-adreno",
    "adreno-730": "gpu-adreno",
    "mali-g77": "gpu-mali-valhall",
    "mali-g78": "gpu-mali-valhall",
    "tesla-v100-sxm2-16gb": "gpu-cuda-sm_70",
    "nvidia-a100-sxm4-40gb": "gpu-cuda-sm_80",
    "nvidia-geforce-rtx-3090": "gpu-cuda-sm_80",
}

# A map from CPU ABI to IREE's benchmark target architecture.
CPU_ABI_TO_TARGET_ARCH_MAP = {
    "arm64-v8a":
        common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
    "x86_64-cascadelake":
        common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
}

# A map from GPU name to IREE's benchmark target architecture.
GPU_NAME_TO_TARGET_ARCH_MAP = {
    "adreno-640":
        common_definitions.DeviceArchitecture.QUALCOMM_ADRENO,
    "adreno-650":
        common_definitions.DeviceArchitecture.QUALCOMM_ADRENO,
    "adreno-660":
        common_definitions.DeviceArchitecture.QUALCOMM_ADRENO,
    "adreno-730":
        common_definitions.DeviceArchitecture.QUALCOMM_ADRENO,
    "mali-g77":
        common_definitions.DeviceArchitecture.ARM_VALHALL,
    "mali-g78":
        common_definitions.DeviceArchitecture.ARM_VALHALL,
    "tesla-v100-sxm2-16gb":
        common_definitions.DeviceArchitecture.NVIDIA_PASCAL,
    "nvidia-a100-sxm4-40gb":
        common_definitions.DeviceArchitecture.NVIDIA_AMPERE,
    "nvidia-geforce-rtx-3090":
        common_definitions.DeviceArchitecture.NVIDIA_AMPERE,
}


@dataclasses.dataclass
class DriverInfo:
  """An object describing a IREE HAL driver.

  It includes the following characteristics:
  - pretty_name: the pretty name, e.g., 'IREE-LLVM-CPU'
  - device_type: the targeted device type, e.g., 'CPU'
  - driver_name: runtime driver flag, e.g., 'local-task'
  - loader_name: executable loader name, if used
  """

  pretty_name: str
  device_type: str
  driver_name: str
  loader_name: str


# A map for IREE driver names. This allows us to normalize driver names like
# mapping to more friendly ones and detach to keep driver names used in
# benchmark presentation stable.
IREE_DRIVERS_INFOS = {
    "iree-llvm-cpu":
        DriverInfo("IREE-LLVM-CPU", "CPU", "local-task", "embedded-elf"),
    "iree-llvm-cpu-sync":
        DriverInfo("IREE-LLVM-CPU-Sync", "CPU", "local-sync", "embedded-elf"),
    "iree-vmvx":
        DriverInfo("IREE-VMVX", "CPU", "local-task", "vmvx-module"),
    "iree-vmvx-sync":
        DriverInfo("IREE-VMVX-Sync", "CPU", "local-sync", "vmvx-module"),
    "iree-vulkan":
        DriverInfo("IREE-Vulkan", "GPU", "vulkan", ""),
    "iree-cuda":
        DriverInfo("IREE-CUDA", "GPU", "cuda", ""),
}

IREE_PRETTY_NAME_TO_DRIVER_NAME = {
    v.pretty_name: k for k, v in IREE_DRIVERS_INFOS.items()
}


def execute_cmd(args: Sequence[Any],
                verbose: bool = False,
                **kwargs) -> subprocess.CompletedProcess:
  """Executes a command and returns the completed process.

  A thin wrapper around subprocess.run that sets some useful defaults and
  optionally prints out the command being run.

  Raises:
    CalledProcessError if the command fails.
  """
  if verbose:
    print(f"cmd: {args}")
  try:
    return subprocess.run(args, check=True, text=True, **kwargs)
  except subprocess.CalledProcessError as exc:
    print((f"\n\nThe following command failed:\n\n{args}"
           f"\n\nReturn code: {exc.returncode}\n\n"))
    if exc.stdout:
      print(f"Stdout:\n\n{exc.stdout}\n\n")
    if exc.stderr:
      print(f"Stderr:\n\n{exc.stderr}\n\n")
    raise exc


def execute_cmd_and_get_output(args: Sequence[Any],
                               verbose: bool = False,
                               **kwargs) -> Tuple[str, str]:
  """Executes a command and returns its stdout and stderr

  Same as execute_cmd except captures stdout and stderr.
  """
  exc = execute_cmd(args,
                    verbose=verbose,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **kwargs)
  return exc.stdout.strip(), exc.stderr.strip()


def execute_cmd_and_get_stdout(args: Sequence[Any],
                               verbose: bool = False,
                               **kwargs) -> str:
  """Executes a command and returns its stdout.

  Same as execute_cmd except captures stdout (and not stderr).
  """
  stdout, _ = execute_cmd_and_get_output(args, verbose=verbose, **kwargs)
  return stdout


def get_git_commit_hash(commit: str) -> str:
  return execute_cmd_and_get_stdout(['git', 'rev-parse', commit],
                                    cwd=pathlib.Path(__file__).resolve().parent)


def get_iree_benchmark_module_arguments(
    results_filename: str,
    driver_info: DriverInfo,
    benchmark_min_time: Optional[float] = None):
  """Returns the common arguments to run iree-benchmark-module."""

  if driver_info.loader_name == "vmvx-module":
    # VMVX is very unoptimized for now and can take a long time to run.
    # Decrease the repetition for it until it's reasonably fast.
    repetitions = 3
  else:
    repetitions = 10

  cmd = [
      "--time_unit=ns",
      "--benchmark_format=json",
      "--benchmark_out_format=json",
      f"--benchmark_out={results_filename}",
      "--print_statistics=true",
  ]
  if benchmark_min_time:
    cmd.extend([
        f"--benchmark_min_time={benchmark_min_time}",
    ])
  else:
    cmd.extend([
        f"--benchmark_repetitions={repetitions}",
    ])

  return cmd


def wait_for_iree_benchmark_module_start(process: subprocess.Popen,
                                         verbose: bool = False) -> None:
  """Wait for the start of iree-benchmark module; otherwise will see connection
  failure when opening the catpure tool."""

  while True:
    line = process.stdout.readline()  # pytype: disable=attribute-error
    if line == "" and process.poll() is not None:  # Process completed
      raise ValueError("Cannot find benchmark result line in the log!")
    if verbose:
      print(line.strip())
    # Result available
    if re.match(r"^BM_.+/real_time", line) is not None:
      break


class PlatformType(Enum):
  ANDROID = "Android"
  LINUX = "Linux"


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
  """An object describing a device.

  It includes the following characteristics:
  - platform_type: the OS platform, e.g., 'Android'
  - model: the product model, e.g., 'Pixel-4'
  - cpu_abi: the CPU ABI, e.g., 'arm64-v8a', 'x86_64'
  - cpu_uarch: the CPU microarchitecture, e.g., 'CascadeLake'
  - cpu_features: the detailed CPU features, e.g., ['fphp', 'sve']
  - gpu_name: the GPU name, e.g., 'Mali-G77'
  """

  platform_type: PlatformType
  model: str
  cpu_abi: str
  cpu_uarch: Optional[str]
  cpu_features: Sequence[str]
  gpu_name: str

  def __str__(self):
    features = ", ".join(self.cpu_features)
    params = [
        f"model='{self.model}'",
        f"cpu_abi='{self.cpu_abi}'",
        f"cpu_uarch='{self.cpu_uarch}'",
        f"gpu_name='{self.gpu_name}'",
        f"cpu_features=[{features}]",
    ]
    params = ", ".join(params)
    return f"{self.platform_type.value} device <{params}>"

  def get_iree_cpu_arch_name(self,
                             use_legacy_name: bool = False) -> Optional[str]:
    name = self.cpu_abi.lower()
    if self.cpu_uarch:
      name += f"-{self.cpu_uarch.lower()}"

    if use_legacy_name:
      return CPU_ABI_TO_LEGACY_TARGET_ARCH_MAP.get(name)

    arch = CPU_ABI_TO_TARGET_ARCH_MAP.get(name)
    # TODO(#11076): Return common_definitions.DeviceArchitecture instead after
    # removing the legacy path.
    return None if arch is None else str(arch)

  def get_iree_gpu_arch_name(self,
                             use_legacy_name: bool = False) -> Optional[str]:
    name = self.gpu_name.lower()

    if use_legacy_name:
      return GPU_NAME_TO_LEGACY_TARGET_ARCH_MAP.get(name)

    arch = GPU_NAME_TO_TARGET_ARCH_MAP.get(name)
    return None if arch is None else str(arch)

  def get_detailed_cpu_arch_name(self) -> str:
    """Returns the detailed architecture name."""

    if self.cpu_abi == "arm64-v8a":
      return self.__get_arm_cpu_arch_revision()
    if self.cpu_abi == "x86_64":
      return self.__get_x86_detailed_cpu_arch_name()
    raise ValueError("Unrecognized CPU ABI; need to update the list")

  def to_json_object(self) -> Dict[str, Any]:
    return {
        "platform_type": self.platform_type.value,
        "model": self.model,
        "cpu_abi": self.cpu_abi,
        "cpu_uarch": self.cpu_uarch if self.cpu_uarch else "",
        "cpu_features": self.cpu_features,
        "gpu_name": self.gpu_name,
    }

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    cpu_uarch = json_object.get("cpu_uarch")
    return DeviceInfo(PlatformType(json_object["platform_type"]),
                      json_object["model"], json_object["cpu_abi"],
                      None if cpu_uarch == "" else cpu_uarch,
                      json_object["cpu_features"], json_object["gpu_name"])

  def __get_x86_detailed_cpu_arch_name(self) -> str:
    """Returns the x86 architecture with microarchitecture name."""

    if not self.cpu_uarch:
      return self.cpu_abi

    return f"{self.cpu_abi}-{self.cpu_uarch}"

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


@dataclasses.dataclass(frozen=True)
class BenchmarkInfo:
  """An object describing the current benchmark.

  It includes the following benchmark characteristics:
  - name: the benchmark name
  - model_name: the model name, e.g., 'MobileNetV2'
  - model_tags: a list of tags used to describe additional model information,
      e.g., ['imagenet']
  - model_source: the source of the model, e.g., 'TensorFlow'
  - bench_mode: a list of tags for benchmark mode,
      e.g., ['1-thread', 'big-core', 'full-inference']
  - compile_tags: an optional list of tags to describe the compile configs,
      e.g., ['fuse-padding']
  - runner: which runner is used for benchmarking, e.g., 'iree_vulkan', 'tflite'
  - device_info: an DeviceInfo object describing the device where benchmarks run
  """

  name: str
  model_name: str
  model_tags: Sequence[str]
  model_source: str
  bench_mode: Sequence[str]
  driver_info: DriverInfo
  device_info: DeviceInfo
  compile_tags: Optional[Sequence[str]] = None
  run_config_id: Optional[str] = None

  def __str__(self):
    return self.name

  @classmethod
  def build_with_legacy_name(cls, model_name: str, model_tags: Sequence[str],
                             model_source: str, bench_mode: Sequence[str],
                             driver_info: DriverInfo, device_info: DeviceInfo):
    """Build legacy name by combining the components of the BenchmarkInfo.

    This is the legacy way to construct the name and still used as primary key
    in the legacy benchmark system. It's deprecated and the new benchmark suites
    use a human-defined name which can be more concise.
    """
    # TODO(#11076): Remove when we drop the legacy path in
    # BenchmarkDriver.__get_benchmark_info_from_case

    # Get the target architecture and better driver name depending on the runner.
    target_arch = None
    if driver_info.device_type == 'GPU':
      target_arch = "GPU-" + device_info.gpu_name
    elif driver_info.device_type == 'CPU':
      target_arch = "CPU-" + device_info.get_detailed_cpu_arch_name()
    else:
      raise ValueError(f"Unrecognized device type '{driver_info.device_type}' "
                       f"of the driver '{driver_info.pretty_name}'")

    if model_tags:
      tags = ",".join(model_tags)
      model_part = f"{model_name} [{tags}] ({model_source})"
    else:
      model_part = f"{model_name} ({model_source})"
    device_part = f"{device_info.model} ({target_arch})"

    mode_tags = ",".join(bench_mode)
    name = (f"{model_part} {mode_tags} with {driver_info.pretty_name} "
            f"@ {device_part}")

    return cls(name=name,
               model_name=model_name,
               model_tags=model_tags,
               model_source=model_source,
               bench_mode=bench_mode,
               driver_info=driver_info,
               device_info=device_info)

  def to_json_object(self) -> Dict[str, Any]:
    return {
        "name": self.name,
        "model_name": self.model_name,
        "model_tags": self.model_tags,
        "model_source": self.model_source,
        "bench_mode": self.bench_mode,
        "compile_tags": self.compile_tags,
        # Get the "iree-*" driver name from the DriverInfo.
        "runner": IREE_PRETTY_NAME_TO_DRIVER_NAME[self.driver_info.pretty_name],
        "device_info": self.device_info.to_json_object(),
        "run_config_id": self.run_config_id
    }

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    driver_info = IREE_DRIVERS_INFOS.get(json_object["runner"])
    if not driver_info:
      raise ValueError(f"Unrecognized runner: {json_object['runner']}")

    return BenchmarkInfo(name=json_object["name"],
                         model_name=json_object["model_name"],
                         model_tags=json_object["model_tags"],
                         model_source=json_object["model_source"],
                         bench_mode=json_object["bench_mode"],
                         compile_tags=json_object.get("compile_tags"),
                         driver_info=driver_info,
                         device_info=DeviceInfo.from_json_object(
                             json_object["device_info"]),
                         run_config_id=json_object.get("run_config_id"))


@dataclasses.dataclass(frozen=True)
class BenchmarkLatency:
  """Stores latency statistics for a benchmark run."""
  mean: int
  median: int
  stddev: int
  unit: str

  def to_json_object(self) -> Dict[str, Any]:
    return dataclasses.asdict(self)

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return BenchmarkLatency(**json_object)


def _get_google_benchmark_latencies(
    benchmark_json: Dict[str,
                         Any]) -> Tuple[BenchmarkLatency, BenchmarkLatency]:
  """Returns the Google Benchmark aggregate latencies.

    Args:
      benchmark_json: The JSON string or object returned by Google Benchmark.

    Returns:
      Real time and CPU time BenchmarkLatency.
    """
  real_time_object: Dict[str, Any] = dict(unit="ns")
  cpu_time_object: Dict[str, Any] = dict(unit="ns")
  metrics = ["mean", "median", "stddev"]
  for case in benchmark_json["benchmarks"]:
    if any(case["name"].endswith(f"real_time_{m}") for m in metrics):
      if case["time_unit"] != "ns":
        raise ValueError(f"Expected ns as time unit")
      metric = case["name"].split("_")[-1]
      real_time_object[metric] = int(round(case["real_time"]))
      cpu_time_object[metric] = int(round(case["cpu_time"]))

  # from_json_object implicitly validates that all metrics were found.
  real_time = BenchmarkLatency.from_json_object(real_time_object)
  cpu_time = BenchmarkLatency.from_json_object(cpu_time_object)
  return real_time, cpu_time


@dataclasses.dataclass(frozen=True)
class BenchmarkMemory:
  """Stores memory statistics for a benchmark run."""
  peak: int
  allocated: int
  freed: int
  live: int
  unit: str

  def to_json_object(self) -> Dict[str, int]:
    return dataclasses.asdict(self)

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return BenchmarkMemory(**json_object)


def _get_iree_memory_statistics(benchmark_stderr: str,
                                device: str) -> BenchmarkMemory:
  """Extracts IREE's memory statistics for a given device."""
  # The memory statistics for each device are listed on their own line.
  pattern = (rf"{device}:"
             r"\s*(?P<peak>\d+)B peak /"
             r"\s*(?P<allocated>\d+)B allocated /"
             r"\s*(?P<freed>\d+)B freed /"
             r"\s*(?P<live>\d+)B live")
  match = re.search(pattern, benchmark_stderr)
  if match is None:
    raise ValueError(
        f"Unable to find memory statistics in '{benchmark_stderr}'")
  return BenchmarkMemory(
      peak=int(match["peak"]),
      allocated=int(match["allocated"]),
      freed=int(match["freed"]),
      live=int(match["live"]),
      unit="bytes",
  )


@dataclasses.dataclass(frozen=True)
class BenchmarkMetrics(object):
  """An object describing the results from a single benchmark.

  - real_time: the real time latency statistics returned by the benchmarking
      framework.
  - cpu_time: the cpu time latency statistics returned by the benchmarking
      framework.
  - host_memory: the host memory statistics returned by the benchmarking
      framework.
  - device_memory: the device memory statistics returned by the benchmarking
      framework.
  - raw_data: additional JSON-compatible raw results returned by the
      benchmarking framework.
  """
  real_time: BenchmarkLatency
  cpu_time: BenchmarkLatency
  host_memory: BenchmarkMemory
  device_memory: BenchmarkMemory
  raw_data: Dict[str, Any]

  def to_json_object(self) -> Dict[str, Any]:
    return {
        "real_time": self.real_time.to_json_object(),
        "cpu_time": self.cpu_time.to_json_object(),
        "host_memory": self.host_memory.to_json_object(),
        "device_memory": self.device_memory.to_json_object(),
        "raw_data": self.raw_data,
    }

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return BenchmarkMetrics(
        real_time=BenchmarkLatency.from_json_object(json_object["real_time"]),
        cpu_time=BenchmarkLatency.from_json_object(json_object["cpu_time"]),
        host_memory=BenchmarkMemory.from_json_object(
            json_object["host_memory"]),
        device_memory=BenchmarkMemory.from_json_object(
            json_object["device_memory"]),
        raw_data=json_object["raw_data"],
    )


def parse_iree_benchmark_metrics(benchmark_stdout: str,
                                 benchmark_stderr: str) -> BenchmarkMetrics:
  """Extract benchmark metrics from the output of iree-benchmark-module.

  Args:
    benchmark_stdout: The stdout of iree-benchmark-module with
      --benchmark_format=json.
    benchmark_stdout: The stderr of iree-benchmark-module with
      --print_statistics=true.

  Returns:
    A populated BenchmarkMetrics dataclass.
  """
  benchmark_json = json.loads(benchmark_stdout)
  real_time, cpu_time = _get_google_benchmark_latencies(benchmark_json)
  return BenchmarkMetrics(
      real_time=real_time,
      cpu_time=cpu_time,
      host_memory=_get_iree_memory_statistics(benchmark_stderr, "HOST_LOCAL"),
      device_memory=_get_iree_memory_statistics(benchmark_stderr,
                                                "DEVICE_LOCAL"),
      raw_data=benchmark_json,
  )


@dataclasses.dataclass(frozen=True)
class BenchmarkRun(object):
  """An object describing a single run of the benchmark binary.

  - info: a BenchmarkInfo object describing the benchmark setup.
  - metrics: a BenchmarkMetrics object containing the results of the benchmark.
  """
  info: BenchmarkInfo
  metrics: BenchmarkMetrics

  def to_json_object(self) -> Dict[str, Any]:
    return {
        "info": self.info.to_json_object(),
        "metrics": self.metrics.to_json_object(),
    }

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return BenchmarkRun(
        BenchmarkInfo.from_json_object(json_object["info"]),
        BenchmarkMetrics.from_json_object(json_object["metrics"]),
    )


class BenchmarkResults(object):
  """An object describing a set of benchmarks for one particular commit.

    It contains the following fields:
    - commit: the commit SHA for this set of benchmarks.
    - benchmarks: a list of BenchmarkRun objects
    """

  def __init__(self):
    self.commit: str = "<unknown>"
    self.benchmarks: List[BenchmarkRun] = []

  def set_commit(self, commit: str):
    self.commit = commit

  def merge(self, other):
    if self.commit != other.commit:
      raise ValueError("Inconsistent pull request commit")
    self.benchmarks.extend(other.benchmarks)

  def to_json_str(self) -> str:
    json_object = {"commit": self.commit, "benchmarks": []}
    json_object["benchmarks"] = [b.to_json_object() for b in self.benchmarks]
    return json.dumps(json_object, indent=2)

  @staticmethod
  def from_json_str(json_str: str):
    json_object = json.loads(json_str)
    results = BenchmarkResults()
    results.set_commit(json_object["commit"])
    results.benchmarks = [
        BenchmarkRun.from_json_object(b) for b in json_object["benchmarks"]
    ]
    return results


@dataclasses.dataclass(frozen=True)
class CompilationInfo(object):
  name: str
  model_name: str
  model_tags: Tuple[str]
  model_source: str
  target_arch: str
  compile_tags: Tuple[str]
  gen_config_id: Optional[str] = None

  def __str__(self):
    return self.name

  @classmethod
  def build_with_legacy_name(cls, model_name: str, model_tags: Sequence[str],
                             model_source: str, target_arch: str,
                             compile_tags: Sequence[str]):
    """Build legacy name by combining the components of the CompilationInfo.

    This is the legacy way to construct the name and still used as primary key
    in the legacy benchmark system. It's deprecated and the new benchmark suites
    use a human-defined name which can be more concise.
    """
    # TODO(#11076): Remove when we drop
    # collect_compilation_statistics.get_module_map_from_benchmark_suite
    if model_tags:
      tags = ",".join(model_tags)
      model_part = f"{model_name} [{tags}] ({model_source})"
    else:
      model_part = f"{model_name} ({model_source})"
    compile_tags_str = ",".join(compile_tags)
    name = f"{model_part} {target_arch} {compile_tags_str}"
    return cls(name=name,
               model_name=model_name,
               model_tags=tuple(model_tags),
               model_source=model_source,
               target_arch=target_arch,
               compile_tags=tuple(compile_tags))

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return CompilationInfo(name=json_object["name"],
                           model_name=json_object["model_name"],
                           model_tags=tuple(json_object["model_tags"]),
                           model_source=json_object["model_source"],
                           target_arch=json_object["target_arch"],
                           compile_tags=tuple(json_object["compile_tags"]),
                           gen_config_id=json_object.get("gen_config_id"))


@dataclasses.dataclass(frozen=True)
class ModuleComponentSizes(object):
  file_bytes: int
  vm_component_bytes: int
  const_component_bytes: int
  total_dispatch_component_bytes: int

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return ModuleComponentSizes(**json_object)


@dataclasses.dataclass(frozen=True)
class IRStatistics(object):
  # Number of cmd.dispatch ops in IR.
  stream_dispatch_count: int

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return IRStatistics(**json_object)


@dataclasses.dataclass(frozen=True)
class CompilationStatistics(object):
  compilation_info: CompilationInfo
  # Module file and component sizes.
  module_component_sizes: ModuleComponentSizes
  # Module compilation time in ms.
  compilation_time_ms: int
  # IR-level statistics
  ir_stats: IRStatistics

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return CompilationStatistics(
        compilation_info=CompilationInfo.from_json_object(
            json_object["compilation_info"]),
        module_component_sizes=ModuleComponentSizes.from_json_object(
            json_object["module_component_sizes"]),
        compilation_time_ms=json_object["compilation_time_ms"],
        ir_stats=IRStatistics.from_json_object(json_object["ir_stats"]))


@dataclasses.dataclass(frozen=True)
class CompilationResults(object):
  commit: str
  compilation_statistics: Sequence[CompilationStatistics]

  @staticmethod
  def from_json_object(json_object: Dict[str, Any]):
    return CompilationResults(
        commit=json_object["commit"],
        compilation_statistics=[
            CompilationStatistics.from_json_object(obj)
            for obj in json_object["compilation_statistics"]
        ])
