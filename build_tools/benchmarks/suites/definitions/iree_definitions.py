## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Classes for IREE benchmark definitions."""

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import List

from . import common_definitions


class TargetBackend(Enum):
  """IREE target backend."""
  LLVM_CPU = "llvm-cpu"
  CUDA = "cuda"
  ROCM = "rocm"
  VMVX = "vmvx"
  METAL_SPIRV = "metal-spirv"
  VULKAN_SPIRV = "vulkan-spirv"


class RuntimeLoader(Enum):
  """IREE runtime loader."""
  EMBEDDED_ELF = "embedded-elf"
  VMVX_MODULE = "vmvx-module"
  SYSTEM_LIBRARY = "system-library"


class RuntimeDriver(Enum):
  """IREE runtime driver."""
  LOCAL_SYNC = "local-sync"
  LOCAL_TASK = "local-task"
  CUDA = "cuda"
  VULKAN = "vulkan"


@dataclass(frozen=True)
class CompileTarget(object):
  """Describes a target device to build for."""
  target_architecture: common_definitions.DeviceArchitecture
  target_platform: common_definitions.DevicePlatform
  target_backend: TargetBackend


@dataclass(frozen=True)
class CompileConfig(object):
  """Describes the options to build a module."""
  id: str
  tags: List[str]
  compile_targets: List[CompileTarget]
  extra_flags: List[str] = dataclasses.field(default_factory=list)


@dataclass(frozen=True)
class RunConfig(object):
  """Describes the options to run a module."""
  id: str
  tags: List[str]
  loader: RuntimeLoader
  driver: RuntimeDriver
  benchmark_tool: str
  extra_flags: List[str] = dataclasses.field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkCompileSpec(object):
  """Describes a compile target to generate the module."""
  compile_config: CompileConfig
  model: common_definitions.Model


@dataclass(frozen=True)
class BenchmarkRunSpec(object):
  """Describes a run target to be benchmarked."""
  compile_spec: BenchmarkCompileSpec
  run_config: RunConfig
  target_device_spec: common_definitions.DeviceSpec
  input_data: common_definitions.ModelInputData
