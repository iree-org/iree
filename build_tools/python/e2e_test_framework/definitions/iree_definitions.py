## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Classes for IREE compilation and run definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import List
import dataclasses

from e2e_test_framework.definitions import common_definitions
from e2e_test_framework import serialization


class TargetBackend(Enum):
  """IREE target backend."""
  LLVM_CPU = "llvm-cpu"
  CUDA = "cuda"
  ROCM = "rocm"
  VMVX = "vmvx"
  METAL_SPIRV = "metal-spirv"
  VULKAN_SPIRV = "vulkan-spirv"


class TargetABI(Enum):
  VMVX = "vmvx"
  LINUX_GNU = "linux-gnu"
  LINUX_ANDROID29 = "linux-android29"
  LINUX_ANDROID31 = "linux-android31"


class RuntimeLoader(Enum):
  """IREE runtime loader."""
  # For target that doesn't support loader configuration.
  NONE = "none"
  EMBEDDED_ELF = "embedded-elf"
  VMVX_MODULE = "vmvx-module"
  SYSTEM_LIBRARY = "system-library"


class RuntimeDriver(Enum):
  """IREE runtime driver."""
  LOCAL_SYNC = "local-sync"
  LOCAL_TASK = "local-task"
  CUDA = "cuda"
  VULKAN = "vulkan"


@serialization.serializable
@dataclass(frozen=True)
class CompileTarget(object):
  """Describes a target device to build for."""
  target_backend: TargetBackend
  target_architecture: common_definitions.DeviceArchitecture
  target_abi: TargetABI


@serialization.serializable(type_key="iree_compile_configs")
@dataclass(frozen=True)
class CompileConfig(object):
  """Describes the options to build a module."""
  id: str
  tags: List[str]
  compile_targets: List[CompileTarget]
  extra_flags: List[str] = dataclasses.field(default_factory=list)


@serialization.serializable(type_key="iree_module_execution_configs")
@dataclass(frozen=True)
class ModuleExecutionConfig(object):
  """Describes the options to run a module."""
  id: str
  tags: List[str]
  loader: RuntimeLoader
  driver: RuntimeDriver
  extra_flags: List[str] = dataclasses.field(default_factory=list)


class MLIRDialectType(Enum):
  """Imported MLIR dialect type."""
  LINALG = "linalg"
  TOSA = "tosa"
  MHLO = "mhlo"


MODEL_SOURCE_TO_DIALECT_TYPE_MAP = {
    common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR:
        MLIRDialectType.LINALG,
    common_definitions.ModelSourceType.EXPORTED_TFLITE:
        MLIRDialectType.TOSA,
    common_definitions.ModelSourceType.EXPORTED_TF:
        MLIRDialectType.MHLO,
}


@serialization.serializable
@dataclass(frozen=True)
class ImportedModel(object):
  """Describes an imported MLIR model."""
  model: common_definitions.Model
  dialect_type: MLIRDialectType

  @staticmethod
  def from_model(model: common_definitions.Model):
    # Currently we assume the model source type and its imported dialect is an
    # 1-1 mapping.
    return ImportedModel(
        model=model,
        dialect_type=MODEL_SOURCE_TO_DIALECT_TYPE_MAP[model.source_type])


@serialization.serializable
@dataclass(frozen=True)
class ModuleGenerationConfig(object):
  """Describes a compile target to generate the module."""
  imported_model: ImportedModel
  compile_config: CompileConfig


@serialization.serializable
@dataclass(frozen=True)
class E2EModelRunConfig(object):
  """Describes an e2e run."""
  module_generation_config: ModuleGenerationConfig
  module_execution_config: ModuleExecutionConfig
  target_device_spec: common_definitions.DeviceSpec
  input_data: common_definitions.ModelInputData
