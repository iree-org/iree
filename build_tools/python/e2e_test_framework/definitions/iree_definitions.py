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
    return unique_ids.hash_composite_id([self.model.id, self.import_config.id])

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
  # Full list of flags to compile with, derived from sub-components, with
  # unmaterialized placeholders. Allows the compile flags to be persisted and
  # decouple from the generation code. Also serves as useful information in the
  # serialized JSON.
  compile_flags: List[str]

  def composite_id(self):
    return unique_ids.hash_composite_id(
        [self.imported_model.composite_id(), self.compile_config.id])

  def materialize_compile_flags(self):
    """Materialize flags with dependent values."""
    return self.compile_flags

  @staticmethod
  def with_flag_generation(imported_model: ImportedModel,
                           compile_config: CompileConfig):
    return ModuleGenerationConfig(
        imported_model=imported_model,
        compile_config=compile_config,
        compile_flags=_generate_compile_flags(
            compile_config, imported_model.import_config.dialect_type))


@serialization.serializable
@dataclass(frozen=True)
class E2EModelRunConfig(object):
  """Describes an e2e run."""
  module_generation_config: ModuleGenerationConfig
  module_execution_config: ModuleExecutionConfig
  target_device_spec: common_definitions.DeviceSpec
  input_data: common_definitions.ModelInputData

  def composite_id(self):
    return unique_ids.hash_composite_id([
        self.module_generation_config.composite_id(),
        self.module_execution_config.id, self.target_device_spec.id,
        self.input_data.id
    ])


def _generate_compile_flags(compile_config: CompileConfig,
                            dialect_type: MLIRDialectType) -> List[str]:
  if len(compile_config.compile_targets) != 1:
    raise ValueError(f"Only one compile target is supported. Got:"
                     f" {compile_config.compile_targets}")

  compile_target = compile_config.compile_targets[0]
  flags = [
      f"--iree-hal-target-backends={compile_target.target_backend.value}",
      f"--iree-input-type={dialect_type.value}"
  ]
  flags += _generate_compile_target_flags(compile_target)
  flags += compile_config.extra_flags
  return flags


def _generate_compile_target_flags(target: CompileTarget) -> List[str]:
  arch_info = target.target_architecture
  if target.target_backend == TargetBackend.VULKAN_SPIRV:
    return [
        f"--iree-vulkan-target-triple={arch_info.architecture}-unknown-{target.target_abi.value}",
    ]

  if arch_info.architecture == "x86_64":
    flags = [
        f"--iree-llvm-target-triple=x86_64-unknown-{target.target_abi.value}",
        f"--iree-llvm-target-cpu={arch_info.microarchitecture.lower()}"
    ]
  elif arch_info.architecture == "riscv_64":
    flags = [
        f"--iree-llvm-target-triple=riscv64-pc-{target.target_abi.value}",
        "--iree-llvm-target-cpu=generic-rv64", "--iree-llvm-target-abi=lp64d",
        "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v",
        "--riscv-v-fixed-length-vector-lmul-max=8"
    ]
  elif arch_info.architecture == "riscv_32":
    # TODO(llvm-project/60463): Replace 'zve32f' with 'zve32x'.
    flags = [
        f"--iree-llvm-target-triple=riscv32-pc-{target.target_abi.value}",
        "--iree-llvm-target-cpu=generic-rv32", "--iree-llvm-target-abi=ilp32",
        "--iree-llvm-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f",
        "--riscv-v-fixed-length-vector-lmul-max=8"
    ]
  elif arch_info.architecture == "armv8.2-a":
    flags = [
        f"--iree-llvm-target-triple=aarch64-none-{target.target_abi.value}",
    ]
  elif arch_info.architecture == "cuda":
    if target.target_abi != TargetABI.LINUX_GNU:
      raise ValueError(
          f"Unsupported target ABI for CUDA backend: `{target.target_abi}`")
    flags = [
        f"--iree-hal-cuda-llvm-target-arch={arch_info.microarchitecture}",
    ]
  elif arch_info.architecture == "vmvx":
    flags = []
  else:
    raise ValueError(f"Unsupported architecture: '{arch_info.architecture}'")
  return flags
