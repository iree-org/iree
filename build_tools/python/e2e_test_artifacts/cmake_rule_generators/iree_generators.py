## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake rules to build IREE artifacts."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pathlib

from e2e_test_artifacts.cmake_rule_generators import common_generators
from e2e_test_artifacts import iree_artifacts
from e2e_test_framework.definitions import common_definitions, iree_definitions
import cmake_builder.rules
import e2e_test_artifacts.cmake_rule_generators.utils as cmake_rule_generators_utils


@dataclass
class IreeModelImportRule(cmake_rule_generators_utils.CMakeRule):
  target_name: str
  output_file_path: str
  cmake_rule: Optional[str]

  def get_rule(self) -> str:
    return self.cmake_rule if self.cmake_rule is not None else ""


@dataclass
class IreeModuleCompileRule(cmake_rule_generators_utils.CMakeRule):
  target_name: str
  output_module_path: str
  cmake_rule: str

  def get_rule(self) -> str:
    return self.cmake_rule


def _generate_iree_compile_target_flags(
    target: iree_definitions.CompileTarget) -> List[str]:
  arch_info: common_definitions.ArchitectureInfo = target.target_architecture.value
  if arch_info.architecture == "x86_64":
    flags = [
        f"--iree-llvm-target-triple=x86_64-unknown-{target.target_abi.value}",
        f"--iree-llvm-target-cpu={arch_info.microarchitecture.lower()}"
    ]
  elif arch_info.architecture == "rv64":
    flags = [
        f"--iree-llvm-target-triple=riscv64-pc-{target.target_abi.value}",
        "--iree-llvm-target-cpu=generic-rv64", "--iree-llvm-target-abi=lp64d",
        "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+v",
        "--riscv-v-vector-bits-min=512",
        "--riscv-v-fixed-length-vector-lmul-max=8"
    ]
  elif arch_info.architecture == "rv32":
    flags = [
        f"--iree-llvm-target-triple=riscv32-pc-{target.target_abi.value}",
        "--iree-llvm-target-cpu=generic-rv32", "--iree-llvm-target-abi=ilp32",
        "--iree-llvm-target-cpu-features=+m,+a,+f,+zvl512b,+zve32x",
        "--riscv-v-vector-bits-min=512",
        "--riscv-v-fixed-length-vector-lmul-max=8"
    ]
  elif arch_info.architecture == "adreno":
    flags = [
        f"--iree-vulkan-target-triple=adreno-unknown-{target.target_abi.value}",
    ]
  elif arch_info.architecture == "mali":
    flags = [
        f"--iree-vulkan-target-triple=valhall-unknown-{target.target_abi.value}",
    ]
  elif arch_info.architecture == "armv8.2-a":
    flags = [
        f"--iree-llvm-target-triple=aarch64-none-{target.target_abi.value}",
    ]
  elif arch_info.architecture == "cuda":
    if target.target_abi != iree_definitions.TargetABI.LINUX_GNU:
      raise ValueError(
          f"Unsupported target ABI for CUDA backend: `{target.target_abi}`")
    flags = [
        f"--iree-hal-cuda-llvm-target-arch=sm_80",
    ]
  elif arch_info.architecture == "vmvx":
    flags = []
  else:
    raise ValueError(f"Unsupported architecture: '{arch_info.architecture}'")
  return flags


def _generate_iree_compile_flags(compile_config: iree_definitions.CompileConfig,
                                 mlir_dialect_type: str) -> List[str]:
  if len(compile_config.compile_targets) != 1:
    raise ValueError(f"Only one compile target is supported. Got:"
                     f" {compile_config.compile_targets}")

  compile_target = compile_config.compile_targets[0]
  flags = [
      f"--iree-hal-target-backends={compile_target.target_backend.value}",
      f"--iree-input-type={mlir_dialect_type}"
  ]
  flags.extend(_generate_iree_compile_target_flags(compile_target))
  return flags


def _build_iree_module_compile_rule(
    root_path: pathlib.PurePath, model_import_rule: IreeModelImportRule,
    module_artifact: iree_artifacts.ModuleArtifact) -> IreeModuleCompileRule:

  imported_model = module_artifact.module_generation_config.model
  compile_config = module_artifact.module_generation_config.compile_config

  # Module target: <package_name>_iree-module-<model_id>-<compile_config_id>
  target_name = f"iree-module-{imported_model.source_model.id}-{compile_config.id}"
  output_path = str(root_path / module_artifact.file_path)

  mlir_dialect_type = imported_model.dialect_type.value
  compile_flags = _generate_iree_compile_flags(
      compile_config=compile_config,
      mlir_dialect_type=mlir_dialect_type) + compile_config.extra_flags

  cmake_rule = (f'# Compile the module "{output_path}"\n' +
                cmake_builder.rules.build_iree_bytecode_module(
                    target_name=target_name,
                    src=model_import_rule.output_file_path,
                    module_name=output_path,
                    flags=compile_flags))
  cmake_rule += cmake_builder.rules.build_add_dependencies(
      target="iree-benchmark-suites",
      deps=[cmake_rule_generators_utils.build_target_path(target_name)])

  # TODO(#10155): Dump the compile flags from iree_bytecode_module into a flagfile.

  return IreeModuleCompileRule(target_name=target_name,
                               output_module_path=output_path,
                               cmake_rule=cmake_rule)


def _build_iree_model_import_rule(
    root_path: pathlib.PurePath, source_model_rule: common_generators.ModelRule,
    imported_model_artifact: iree_artifacts.ImportedModelArtifact
) -> IreeModelImportRule:

  model = imported_model_artifact.imported_model.source_model
  if model.source_type == common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR:
    return IreeModelImportRule(target_name=source_model_rule.target_name,
                               output_file_path=source_model_rule.file_path,
                               cmake_rule=None)

  # Import target: <package_name>_iree-import-model-<model_id>
  target_name = f"iree-import-model-{model.id}"

  output_file_path = str(root_path / imported_model_artifact.file_path)

  if model.source_type == common_definitions.ModelSourceType.EXPORTED_TFLITE:
    cmake_rule = (
        f'# Import the TFLite model "{source_model_rule.file_path}"\n' +
        cmake_builder.rules.build_iree_import_tflite_model(
            target_path=cmake_rule_generators_utils.build_target_path(
                target_name),
            source=source_model_rule.file_path,
            output_mlir_file=output_file_path))
  elif model.source_type == common_definitions.ModelSourceType.EXPORTED_TF:
    cmake_rule = (
        f'# Import the Tensorflow model "{source_model_rule.file_path}"\n' +
        cmake_builder.rules.build_iree_import_tf_model(
            target_path=cmake_rule_generators_utils.build_target_path(
                target_name),
            source=source_model_rule.file_path,
            entry_function=model.entry_function,
            output_mlir_file=output_file_path))
  else:
    raise ValueError(
        f"Unsupported source type '{model.source_type}' of the model '{model.id}'."
    )

  cmake_builder.rules.build_add_dependencies(
      target="iree-benchmark-import-models",
      deps=[cmake_rule_generators_utils.build_target_path(target_name)])

  return IreeModelImportRule(target_name=target_name,
                             output_file_path=output_file_path,
                             cmake_rule=cmake_rule)


def generate_rules(
    root_path: pathlib.PurePath,
    iree_model_dir_map: Dict[str, iree_artifacts.ModelDirectory],
    model_rule_map: Dict[str, common_generators.ModelRule]
) -> List[cmake_rule_generators_utils.CMakeRule]:
  """Generates all rules to build IREE artifacts."""

  model_import_rules = []
  module_compile_rules = []
  for model_dir in iree_model_dir_map.values():
    imported_model_artifact = model_dir.imported_model_artifact
    model_rule = model_rule_map[
        imported_model_artifact.imported_model.source_model.id]
    model_import_rule = _build_iree_model_import_rule(
        root_path=root_path,
        source_model_rule=model_rule,
        imported_model_artifact=imported_model_artifact)
    model_import_rules.append(model_import_rule)

    for module_dir in model_dir.module_dir_map.values():
      module_artifact = module_dir.module_artifact
      module_compile_rules.append(
          _build_iree_module_compile_rule(root_path=root_path,
                                          model_import_rule=model_import_rule,
                                          module_artifact=module_artifact))

  return model_import_rules + module_compile_rules
