## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Classes for IREE compilation and run definitions."""

import dataclasses
from dataclasses import dataclass
from enum import Enum
import pathlib
from typing import List, Optional, Sequence

from e2e_test_framework.definitions import common_definitions, utils
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

  def __str__(self):
    return (f"{self.target_architecture}-"
            f"{self.target_abi.name}-"
            f"{self.target_backend.name}").lower()


@serialization.serializable(type_key="iree_compile_configs")
@dataclass(frozen=True)
class CompileConfig(object):
  """Describes the options to build a module."""
  id: str
  name: str
  tags: List[str]
  compile_targets: List[CompileTarget]
  extra_flags: List[str] = dataclasses.field(default_factory=list)

  def __str__(self):
    return self.name

  @classmethod
  def build(cls,
            id: str,
            tags: Sequence[str],
            compile_targets: Sequence[CompileTarget],
            extra_flags: Optional[Sequence[str]] = None):
    target_part = ",".join(str(target) for target in compile_targets)
    tag_part = ",".join(tags)
    # Format: [<target_name>,...][<tag>,...]
    name = f"[{target_part}][{tag_part}]"
    extra_flags = extra_flags or []
    return cls(id=id,
               name=name,
               tags=list(tags),
               compile_targets=list(compile_targets),
               extra_flags=list(extra_flags))


@serialization.serializable(type_key="iree_module_execution_configs")
@dataclass(frozen=True)
class ModuleExecutionConfig(object):
  """Describes the options to run a module."""
  id: str
  name: str
  tags: List[str]
  loader: RuntimeLoader
  driver: RuntimeDriver
  extra_flags: List[str] = dataclasses.field(default_factory=list)

  def __str__(self):
    return self.name

  @classmethod
  def build(cls,
            id: str,
            tags: Sequence[str],
            loader: RuntimeLoader,
            driver: RuntimeDriver,
            extra_flags: Optional[Sequence[str]] = None):
    runtime_part = f"{driver.name}({loader.name})".lower()
    tag_part = ",".join(tags)
    # Format: <driver>(<loader>)[<tag>,...]
    name = f"{runtime_part}[{tag_part}]"
    extra_flags = extra_flags or []
    return cls(id=id,
               name=name,
               tags=list(tags),
               loader=loader,
               driver=driver,
               extra_flags=list(extra_flags))


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


@serialization.serializable(type_key="iree_import_configs")
@dataclass(frozen=True)
class ImportConfig(object):
  """Config to import the model."""
  id: str
  name: str
  tool: ImportTool
  dialect_type: MLIRDialectType
  import_flags: List[str] = dataclasses.field(default_factory=list)

  def __str__(self):
    return self.name

  def materialize_import_flags(self,
                               model: common_definitions.Model) -> List[str]:
    """Materialize flags with dependent values."""
    return utils.substitute_flag_vars(flags=self.import_flags,
                                      ENTRY_FUNCTION=model.entry_function)


DEFAULT_TF_V1_IMPORT_CONFIG = ImportConfig(
    id=unique_ids.IREE_MODEL_IMPORT_TF_V1_DEFAULT,
    name="tf_v1",
    tool=ImportTool.TF_IMPORTER,
    dialect_type=MLIRDialectType.MHLO,
    import_flags=[
        "--output-format=mlir-bytecode", "--tf-import-type=savedmodel_v1",
        r"--tf-savedmodel-exported-names=${ENTRY_FUNCTION}"
    ])

DEFAULT_TF_V2_IMPORT_CONFIG = ImportConfig(
    id=unique_ids.IREE_MODEL_IMPORT_TF_V1_DEFAULT,
    tool=ImportTool.TF_IMPORTER,
    name="tf_v2",
    dialect_type=MLIRDialectType.MHLO,
    import_flags=[
        "--output-format=mlir-bytecode", "--tf-import-type=savedmodel_v2",
        r"--tf-savedmodel-exported-names=${ENTRY_FUNCTION}"
    ])

DEFAULT_TFLITE_IMPORT_CONFIG = ImportConfig(
    id=unique_ids.IREE_MODEL_IMPORT_TFLITE_DEFAULT,
    name="tflite",
    tool=ImportTool.TFLITE_IMPORTER,
    dialect_type=MLIRDialectType.TOSA,
    import_flags=["--output-format=mlir-bytecode"])

DEFAULT_LINALG_MLIR_IMPORT_CONFIG = ImportConfig(
    id=unique_ids.IREE_MODEL_IMPORT_LINALG_MLIR_DEFAULT,
    name="linalg",
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


@serialization.serializable(type_key="iree_imported_models",
                            id_field="composite_id")
@dataclass(frozen=True)
class ImportedModel(object):
  """Describes an imported MLIR model."""
  composite_id: str
  name: str
  model: common_definitions.Model
  import_config: ImportConfig

  def __str__(self):
    return self.name

  @classmethod
  def from_model(cls, model: common_definitions.Model):
    config = MODEL_SOURCE_TO_DEFAULT_IMPORT_CONFIG_MAP.get(model.source_type)
    if config is None:
      raise ValueError(f"Unsupported model source type: {model.source_type}.")

    composite_id = unique_ids.hash_composite_id([model.id, config.id])
    # Format: <model_name>(<import_config_name>)
    name = f"{model}({config})"
    return cls(composite_id=composite_id,
               name=name,
               model=model,
               import_config=config)


# Variable in flags to be replaced with module dir path. The whole path should
# be written in the POSIX format and starts with the variable.
MODULE_DIR_VARIABLE = r"${MODULE_DIR}"


@serialization.serializable(type_key="iree_module_generation_configs",
                            id_field="composite_id")
@dataclass(frozen=True)
class ModuleGenerationConfig(object):
  """Describes a compile target to generate the module."""
  composite_id: str
  name: str
  imported_model: ImportedModel
  compile_config: CompileConfig
  # Full list of flags to compile with, derived from sub-components, with
  # unmaterialized placeholders. Allows the compile flags to be persisted and
  # decouple from the generation code. Also serves as useful information in the
  # serialized JSON.
  compile_flags: List[str]

  def __str__(self):
    return self.name

  def materialize_compile_flags(self, module_dir_path: pathlib.PurePath):
    """Materialize flags with dependent values."""

    def _replace_module_dir_placeholder(value: str) -> str:
      """Replaces ${MODULE_DIR} in a POSIX path and returns the
      platform-dependent path string.
      """
      parts = pathlib.PurePosixPath(value).parts
      if MODULE_DIR_VARIABLE not in parts:
        return value
      if parts[0] != MODULE_DIR_VARIABLE:
        raise ValueError(
            f"'{MODULE_DIR_VARIABLE}' needs to be the head of flag value"
            f" if present, but got '{value}'.")
      # Properly construct the platform-dependent path.
      return str(module_dir_path.joinpath(*parts[1:]))

    return utils.transform_flags(flags=self.compile_flags,
                                 map_funcs=[_replace_module_dir_placeholder])

  @classmethod
  def build(cls, imported_model: ImportedModel, compile_config: CompileConfig):
    composite_id = unique_ids.hash_composite_id(
        [imported_model.composite_id, compile_config.id])
    # Format: <imported_model_name> <compile_config_name>
    name = f"{imported_model} {compile_config}"
    compile_flags = _generate_compile_flags(
        compile_config, imported_model.import_config.dialect_type)
    return cls(composite_id=composite_id,
               name=name,
               imported_model=imported_model,
               compile_config=compile_config,
               compile_flags=compile_flags)


class E2EModelRunTool(Enum):
  """Tool to run a module."""
  IREE_BENCHMARK_MODULE = "iree-benchmark-module"


@serialization.serializable(type_key="iree_e2e_model_run_configs",
                            id_field="composite_id")
@dataclass(frozen=True)
class E2EModelRunConfig(object):
  """Describes an e2e run."""
  composite_id: str
  name: str
  tags: List[str]
  module_generation_config: ModuleGenerationConfig
  module_execution_config: ModuleExecutionConfig
  target_device_spec: common_definitions.DeviceSpec
  input_data: common_definitions.ModelInputData
  # Full list of flags to run with, derived from sub-components, with
  # unmaterialized placeholders. Allows the run flags to be persisted and
  # decouple from the generation code. Also serves as useful information in the
  # serialized JSON.
  run_flags: List[str]
  tool: E2EModelRunTool

  def __str__(self):
    return self.name

  def materialize_run_flags(self, gpu_id: str = "0"):
    """Materialize flags with dependent values."""
    return utils.substitute_flag_vars(flags=self.run_flags, GPU_ID=gpu_id)

  @classmethod
  def build(cls,
            module_generation_config: ModuleGenerationConfig,
            module_execution_config: ModuleExecutionConfig,
            target_device_spec: common_definitions.DeviceSpec,
            input_data: common_definitions.ModelInputData,
            tool: E2EModelRunTool,
            tags: Optional[Sequence[str]] = None):
    composite_id = unique_ids.hash_composite_id([
        module_generation_config.composite_id, module_execution_config.id,
        target_device_spec.id, input_data.id
    ])
    # Format: <module_generation_config_name> <module_execution_config_name> with <input_data_name> @ <target_device_spec_name>
    name = f"{module_generation_config} {module_execution_config} with {input_data} @ {target_device_spec}"
    run_flags = generate_run_flags(
        imported_model=module_generation_config.imported_model,
        input_data=input_data,
        module_execution_config=module_execution_config,
        gpu_id=r"${GPU_ID}")
    tags_list = [] if tags is None else list(tags)
    return cls(composite_id=composite_id,
               name=name,
               tags=tags_list,
               module_generation_config=module_generation_config,
               module_execution_config=module_execution_config,
               target_device_spec=target_device_spec,
               input_data=input_data,
               run_flags=run_flags,
               tool=tool)


def generate_run_flags(imported_model: ImportedModel,
                       input_data: common_definitions.ModelInputData,
                       module_execution_config: ModuleExecutionConfig,
                       gpu_id: str = "0",
                       with_driver: bool = True) -> List[str]:
  """Returns the IREE run module flags of the input model and execution config.
  Args:
    model: source model.
    input_data: model input data.
    module_execution_config: execution config.
    gpu_id: target gpu id, if runs on GPUs.
    with_driver: populate the driver flags if true. False can be used for
      generating flags for some CMake rules with a separate DRIVER arg.
  Returns:
    List of flags.
  """

  model = imported_model.model
  run_flags = [f"--function={model.entry_function}"]
  if input_data != common_definitions.ZEROS_MODEL_INPUT_DATA:
    raise ValueError("Currently only support all-zeros data.")
  run_flags += [f"--input={input_type}=0" for input_type in model.input_types]

  exec_config = module_execution_config
  run_flags += exec_config.extra_flags.copy()
  if with_driver:
    driver = exec_config.driver
    if driver == RuntimeDriver.CUDA:
      run_flags.append(f"--device=cuda://{gpu_id}")
    else:
      run_flags.append(f"--device={driver.value}")

  return run_flags


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
    gpu_arch = arch_info.microarchitecture if len(
        arch_info.microarchitecture) != 0 else arch_info.architecture
    return [
        f"--iree-vulkan-target-triple={gpu_arch}-unknown-{target.target_abi.value}",
    ]

  if arch_info.architecture == "x86_64":
    flags = [
        f"--iree-llvmcpu-target-triple=x86_64-unknown-{target.target_abi.value}",
        f"--iree-llvmcpu-target-cpu={arch_info.microarchitecture.lower()}"
    ]
  elif arch_info.architecture == "riscv_64":
    flags = [
        f"--iree-llvmcpu-target-triple=riscv64-pc-{target.target_abi.value}",
        "--iree-llvmcpu-target-cpu=generic-rv64",
        "--iree-llvmcpu-target-abi=lp64d",
        "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+v",
        "--riscv-v-fixed-length-vector-lmul-max=8"
    ]
  elif arch_info.architecture == "riscv_32":
    # TODO(llvm-project/60463): Replace 'zve32f' with 'zve32x'.
    flags = [
        f"--iree-llvmcpu-target-triple=riscv32-pc-{target.target_abi.value}",
        "--iree-llvmcpu-target-cpu=generic-rv32",
        "--iree-llvmcpu-target-abi=ilp32",
        "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+zvl512b,+zve32f",
        "--riscv-v-fixed-length-vector-lmul-max=8"
    ]
  elif arch_info.architecture == "armv8.2-a":
    flags = [
        f"--iree-llvmcpu-target-triple=aarch64-none-{target.target_abi.value}",
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
