## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from enum import Enum
from typing import List

from definitions.common import DeviceArchitecture, DevicePlatform, Model


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
  compile_targets: List[IreeCompileTarget]
  tags: List[str]
  compile_flags: List[str]


@dataclass(frozen=True)
class IreeRuntimeConfig(object):
  id: str
  loader: IreeRuntimeLoader
  driver: IreeRuntimeDriver
  tags: List[str]
  runtime_flags: List[str]


@dataclass(frozen=True)
class IreeBenchmarkCompileSpec(object):
  compile_config_id: str
  model: Model


@dataclass(frozen=True)
class IreeBenchmarkRunSpec(object):
  benchmark_tool: str
  compile_spec: IreeBenchmarkCompileSpec
  runtime_config_id: str
  target_device_spec_id: str
  input_data_id: str
