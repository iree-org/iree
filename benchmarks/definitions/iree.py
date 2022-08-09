## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from enum import Enum
from typing import List

from benchmarks.definitions.common import DeviceArchitecture, DevicePlatform, DeviceSpec, Model, ModelInputData


class IreeTargetBackend(Enum):
  LLVM_CPU = "llvm-cpu"


class IreeRuntimeLoader(Enum):
  EMBEDDED_ELF = "embedded-elf"


class IreeRuntimeDriver(Enum):
  LOCAL_SYNC = "local-sync"
  LOCAL_TASK = "local-task"


@dataclass(frozen=True)
class IreeCompileTarget(object):
  target_architecture: DeviceArchitecture
  target_platform: DevicePlatform
  target_backend: IreeTargetBackend


@dataclass(frozen=True)
class IreeCompileConfig(object):
  id: str
  tags: List[str]
  compile_targets: List[IreeCompileTarget]
  extra_flags: List[str]


@dataclass(frozen=True)
class IreeRunConfig(object):
  id: str
  tags: List[str]
  loader: IreeRuntimeLoader
  driver: IreeRuntimeDriver
  benchmark_tool: str
  extra_flags: List[str]


@dataclass(frozen=True)
class IreeBenchmarkCompileSpec(object):
  compile_config: IreeCompileConfig
  model: Model


@dataclass(frozen=True)
class IreeBenchmarkRunSpec(object):
  compile_spec: IreeBenchmarkCompileSpec
  run_config: IreeRunConfig
  target_device_spec: DeviceSpec
  input_data: ModelInputData
