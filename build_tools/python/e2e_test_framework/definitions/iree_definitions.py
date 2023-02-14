## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Classes for IREE compilation and run definitions."""

import dataclasses
from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import List, Sequence

from e2e_test_framework.definitions import common_definitions
from e2e_test_framework import serialization, unique_ids


def _hash_composite_id(keys: Sequence[str]) -> str:
  """Computes the composite hash id from string keys.

  String keys are the component ids that compose this composite object. We hash
  the composite id since the id isn't designed to be inspected and insufficient
  to reconstruct the original composite object.

  Args:
    keys: list of string keys.

  Returns:
    Unique hash id.
  """
  return hashlib.sha256(":".join(keys).encode("utf-8")).hexdigest()


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
  # IREE defined OS name for vulkan target. See:
  # compiler/src/iree/compiler/Dialect/Vulkan/IR/VulkanBase.td
  VULKAN_ANDROID30 = "android30"
  VULKAN_ANDROID31 = "android31"


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


class ImportTool(Enum):
  """Iree model import tool."""
  NONE = "none"
  TF_IMPORTER = "iree-import-tf"
  TFLITE_IMPORTER = "iree-import-tflite"


# Value should be the name of an IREE supported input type (--iree-input-type).
class MLIRDialectType(Enum):
  """Imported MLIR dialect type."""
  NONE = "none"
  TOSA = "tosa"
  MHLO = "mhlo"


# Placeholder to be replaced with entry function name when outputting the actual
# flag list.
IMPORT_CONFIG_ENTRY_FUNCTION_PLACEHOLDER = "$ENTRY_FUNCTION_PLACEHOLDER"


@serialization.serializable(type_key="iree_import_configs")
@dataclass(frozen=True)
class ImportConfig(object):
  """Config to import the model."""
  id: str
  tool: ImportTool
  dialect_type: MLIRDialectType
  import_flags: List[str] = dataclasses.field(default_factory=list)

  def materialize_import_flags(self,
                               model: common_definitions.Model) -> List[str]:
    """Materialize flags with dependent values."""
    return [
        flag.replace(IMPORT_CONFIG_ENTRY_FUNCTION_PLACEHOLDER,
                     model.entry_function) for flag in self.import_flags
    ]


DEFAULT_TF_V1_IMPORT_CONFIG = ImportConfig(
    id=unique_ids.IREE_MODEL_IMPORT_TF_V1_DEFAULT,
    tool=ImportTool.TF_IMPORTER,
    dialect_type=MLIRDialectType.MHLO,
    import_flags=[
        "--output-format=mlir-bytecode", "--tf-import-type=savedmodel_v1",
        f"--tf-savedmodel-exported-names={IMPORT_CONFIG_ENTRY_FUNCTION_PLACEHOLDER}"
    ])

DEFAULT_TF_V2_IMPORT_CONFIG = ImportConfig(
    id=unique_ids.IREE_MODEL_IMPORT_TF_V1_DEFAULT,
    tool=ImportTool.TF_IMPORTER,
    dialect_type=MLIRDialectType.MHLO,
    import_flags=[
        "--output-format=mlir-bytecode", "--tf-import-type=savedmodel_v2",
        f"--tf-savedmodel-exported-names={IMPORT_CONFIG_ENTRY_FUNCTION_PLACEHOLDER}"
    ])

DEFAULT_TFLITE_IMPORT_CONFIG = ImportConfig(
    id=unique_ids.IREE_MODEL_IMPORT_TFLITE_DEFAULT,
    tool=ImportTool.TFLITE_IMPORTER,
    dialect_type=MLIRDialectType.TOSA,
    import_flags=["--output-format=mlir-bytecode"])

DEFAULT_LINALG_MLIR_IMPORT_CONFIG = ImportConfig(
    id=unique_ids.IREE_MODEL_IMPORT_LINALG_MLIR_DEFAULT,
    tool=ImportTool.NONE,
    dialect_type=MLIRDialectType.NONE)

MODEL_SOURCE_TO_DEFAULT_IMPORT_CONFIG_MAP = {
    common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR:
        DEFAULT_LINALG_MLIR_IMPORT_CONFIG,
    common_definitions.ModelSourceType.EXPORTED_TFLITE:
        DEFAULT_TFLITE_IMPORT_CONFIG,
    common_definitions.ModelSourceType.EXPORTED_TF_V1:
        DEFAULT_TF_V1_IMPORT_CONFIG,
    common_definitions.ModelSourceType.EXPORTED_TF_V2:
        DEFAULT_TF_V2_IMPORT_CONFIG,
}


@serialization.serializable
@dataclass(frozen=True)
class ImportedModel(object):
  """Describes an imported MLIR model."""
  model: common_definitions.Model
  import_config: ImportConfig

  def composite_id(self):
    return _hash_composite_id([self.model.id, self.import_config.id])

  @staticmethod
  def from_model(model: common_definitions.Model):
    config = MODEL_SOURCE_TO_DEFAULT_IMPORT_CONFIG_MAP.get(model.source_type)
    if config is None:
      raise ValueError(f"Unsupported model source type: {model.source_type}.")

    return ImportedModel(model=model, import_config=config)


@serialization.serializable
@dataclass(frozen=True)
class ModuleGenerationConfig(object):
  """Describes a compile target to generate the module."""
  imported_model: ImportedModel
  compile_config: CompileConfig

  def composite_id(self):
    return _hash_composite_id(
        [self.imported_model.composite_id(), self.compile_config.id])


@serialization.serializable
@dataclass(frozen=True)
class E2EModelRunConfig(object):
  """Describes an e2e run."""
  module_generation_config: ModuleGenerationConfig
  module_execution_config: ModuleExecutionConfig
  target_device_spec: common_definitions.DeviceSpec
  input_data: common_definitions.ModelInputData

  def composite_id(self):
    return _hash_composite_id([
        self.module_generation_config.composite_id(),
        self.module_execution_config.id, self.target_device_spec.id,
        self.input_data.id
    ])
