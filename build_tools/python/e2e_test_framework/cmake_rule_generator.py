## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generator that generates CMake rules from python defined benchmarks.

The rules will build required artifacts to run benchmarks.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence
import os
import pathlib
import urllib.parse

from e2e_test_framework.definitions import common_definitions, iree_definitions
import cmake_builder.rules

# Archive extensions used to pack models.
ARCHIVE_FILE_EXTENSIONS = [".tar", ".gz"]

# CMake variable name to store IREE package name.
PACKAGE_NAME_CMAKE_VARIABLE = "_PACKAGE_NAME"


@dataclass
class ModelRule(object):
  target_name: str
  file_path: str
  cmake_rule: str


@dataclass
class IreeModelImportRule(object):
  target_name: str
  model_id: str
  model_name: str
  output_file_path: str
  mlir_dialect_type: str
  cmake_rule: Optional[str]


@dataclass
class IreeModuleCompileRule(object):
  target_name: str
  output_module_path: str
  cmake_rule: str


def _build_target_path(target_name: str):
  """Returns the full target path by combining the variable of package name and
  the target name.
  """
  return f"${{{PACKAGE_NAME_CMAKE_VARIABLE}}}_{target_name}"


class CommonRuleFactory(object):
  """Generates common cmake rules."""

  def __init__(self, model_artifacts_dir: str):
    """Constructs a CommonRuleFactory.

    Args:
      model_artifacts_dir: root directory to store model files. Can contain
      CMake variable syntax in the path.
    """
    self._model_artifacts_dir = model_artifacts_dir
    self._model_rules = {}

  def add_model_rule(self, model: common_definitions.Model) -> ModelRule:
    """Adds a rule to fetch a model. Reuses the existing rule when possible."""
    if model.id in self._model_rules:
      return self._model_rules[model.id]

    # Model target: <package_name>-model-<model_id>
    target_name = f"model-{model.id}"

    model_url = urllib.parse.urlparse(model.source_url)

    # Drop the archive extensions.
    file_exts = pathlib.PurePath(model_url.path).suffixes
    while len(file_exts) > 0 and file_exts[-1] in ARCHIVE_FILE_EXTENSIONS:
      file_exts.pop()
    model_ext = "".join(file_exts)

    # Model path: <model_artifacts_dir>/<model_id>_<model_name><model_ext>
    model_path = f"{self._model_artifacts_dir}/{model.id}_{model.name}{model_ext}"

    if model_url.scheme == "https":
      cmake_rule = (f'# Fetch the model from "{model.source_url}"\n' +
                    cmake_builder.rules.build_iree_fetch_artifact(
                        target_name=target_name,
                        source_url=model.source_url,
                        output=model_path,
                        unpack=True))
    else:
      raise ValueError("Unsupported model url: {model.source_url}.")

    model_rule = ModelRule(target_name=target_name,
                           file_path=model_path,
                           cmake_rule=cmake_rule)

    self._model_rules[model.id] = model_rule
    return model_rule

  def generate_cmake_rules(self) -> List[str]:
    """Dump all cmake rules in a correct order."""
    return [rule.cmake_rule for rule in self._model_rules.values()]


class IreeRuleFactory(object):
  """Generates IREE benchmark cmake rules."""

  def __init__(self, iree_artifacts_dir):
    """Constructs an IreeRuleFactory.

    Args:
      iree_artifacts_dir: root directory to store generated IREE artifacts. Can
        contain CMake variable syntax in the path.
    """
    self._iree_artifacts_dir = iree_artifacts_dir
    self._import_model_rules = {}
    self._compile_module_rules = {}

  def add_import_model_rule(
      self,
      imported_model: iree_definitions.ImportedModel,
      source_model_rule: ModelRule,
  ) -> IreeModelImportRule:
    """Adds a rule to fetch the model and import into MLIR. Reuses the rule when
    possible."""

    model = imported_model.model
    if model.id in self._import_model_rules:
      return self._import_model_rules[model.id]

    # If the source model is MLIR, no import rule is needed.
    if model.source_type == common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR:
      import_model_rule = IreeModelImportRule(
          target_name=source_model_rule.target_name,
          model_id=model.id,
          model_name=model.name,
          output_file_path=source_model_rule.file_path,
          mlir_dialect_type=imported_model.dialect_type.value,
          cmake_rule=None)
      self._import_model_rules[model.id] = import_model_rule
      return import_model_rule

    # Import target: <package_name>_iree-import-model-<model_id>
    target_name = f"iree-import-model-{model.id}"

    # Imported MLIR path: <iree_artifacts_dir>/<model_id>_<model_name>/<model_name>.mlir
    output_file_path = f"{self._iree_artifacts_dir}/{model.id}_{model.name}/{model.name}.mlir"

    if model.source_type == common_definitions.ModelSourceType.EXPORTED_TFLITE:
      cmake_rule = (
          f'# Import the TFLite model "{source_model_rule.file_path}"\n' +
          cmake_builder.rules.build_iree_import_tflite_model(
              target_path=_build_target_path(target_name),
              source=source_model_rule.file_path,
              output_mlir_file=output_file_path))
    elif model.source_type == common_definitions.ModelSourceType.EXPORTED_TF:
      cmake_rule = (
          f'# Import the Tensorflow model "{source_model_rule.file_path}"\n' +
          cmake_builder.rules.build_iree_import_tf_model(
              target_path=_build_target_path(target_name),
              source=source_model_rule.file_path,
              entry_function=model.entry_function,
              output_mlir_file=output_file_path))
    else:
      raise ValueError(
          f"Unsupported source type '{model.source_type}' of the model '{model.id}'."
      )

    cmake_builder.rules.build_add_dependencies(
        target="iree-benchmark-import-models",
        deps=[_build_target_path(target_name)])

    import_model_rule = IreeModelImportRule(
        target_name=target_name,
        model_id=model.id,
        model_name=model.name,
        output_file_path=output_file_path,
        mlir_dialect_type=imported_model.dialect_type.value,
        cmake_rule=cmake_rule)

    self._import_model_rules[model.id] = import_model_rule
    return import_model_rule

  def add_compile_module_rule(self,
                              compile_config: iree_definitions.CompileConfig,
                              model_import_rule: IreeModelImportRule):
    """Adds a rule to compile a MLIR into a IREE module. Reuses the existing
    rule when possible."""

    model_id = model_import_rule.model_id
    model_name = model_import_rule.model_name

    target_id = f"{model_id}-{compile_config.id}"
    if target_id in self._compile_module_rules:
      return self._compile_module_rules[target_id]

    # Module target: <package_name>_iree-module-<model_id>-<compile_config_id>
    target_name = f"iree-module-{target_id}"

    # Module path: <iree_artifacts_dir>/<model_id>_<model_name>/<compile_config_id>.vmfb
    output_path = os.path.join(self._iree_artifacts_dir,
                               f"{model_id}_{model_name}",
                               f"{compile_config.id}.vmfb")

    compile_flags = self._generate_iree_compile_flags(
        compile_config=compile_config,
        mlir_dialect_type=model_import_rule.mlir_dialect_type
    ) + compile_config.extra_flags

    cmake_rule = (f'# Compile the module "{output_path}"\n' +
                  cmake_builder.rules.build_iree_bytecode_module(
                      target_name=target_name,
                      src=model_import_rule.output_file_path,
                      module_name=output_path,
                      flags=compile_flags))
    cmake_rule += cmake_builder.rules.build_add_dependencies(
        target="iree-benchmark-suites", deps=[_build_target_path(target_name)])
    compile_module_rule = IreeModuleCompileRule(target_name=target_name,
                                                output_module_path=output_path,
                                                cmake_rule=cmake_rule)

    # TODO(#10155): Dump the compile flags from iree_bytecode_module into a flagfile.

    self._compile_module_rules[target_id] = compile_module_rule
    return compile_module_rule

  def generate_cmake_rules(self) -> List[str]:
    """Dump all cmake rules in a correct order."""
    import_model_rules = [
        rule.cmake_rule for rule in self._import_model_rules.values()
    ]
    compile_module_rules = [
        rule.cmake_rule for rule in self._compile_module_rules.values()
    ]
    return import_model_rules + compile_module_rules

  def _generate_iree_compile_flags(
      self, compile_config: iree_definitions.CompileConfig,
      mlir_dialect_type: str) -> List[str]:
    if len(compile_config.compile_targets) != 1:
      raise ValueError(f"Only one compile target is supported. Got:"
                       f" {compile_config.compile_targets}")

    compile_target = compile_config.compile_targets[0]
    flags = [
        f"--iree-hal-target-backends={compile_target.target_backend.value}",
        f"--iree-input-type={mlir_dialect_type}"
    ]
    flags.extend(self._generate_iree_compile_target_flags(compile_target))
    return flags

  def _generate_iree_compile_target_flags(
      self, target: iree_definitions.CompileTarget) -> List[str]:
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


def _generate_iree_rules(
    common_rule_factory: CommonRuleFactory, iree_artifacts_dir: str,
    module_generation_configs: Sequence[iree_definitions.ModuleGenerationConfig]
) -> List[str]:
  iree_rule_factory = IreeRuleFactory(iree_artifacts_dir)
  for module_generation_config in module_generation_configs:
    model = module_generation_config.imported_model.model
    compile_config = module_generation_config.compile_config

    source_model_rule = common_rule_factory.add_model_rule(model)
    import_rule = iree_rule_factory.add_import_model_rule(
        imported_model=module_generation_config.imported_model,
        source_model_rule=source_model_rule)
    iree_rule_factory.add_compile_module_rule(compile_config=compile_config,
                                              model_import_rule=import_rule)

  return iree_rule_factory.generate_cmake_rules()


def generate_rules(
    model_artifacts_dir: str, iree_artifacts_dir: str,
    iree_module_generation_configs: Sequence[
        iree_definitions.ModuleGenerationConfig]
) -> List[str]:
  """Generates cmake rules to build benchmarks.
  
  Args:
    model_artifacts_dir: root directory to store model files. Can contain CMake
      variable syntax in the path.
    iree_artifacts_dir: root directory to store generated IREE artifacts. Can
      contain CMake variable syntax in the path.
    iree_module_generation_configs: compile configs for IREE targets.
  Returns:
    List of CMake rules.
  """
  common_rule_factory = CommonRuleFactory(model_artifacts_dir)
  iree_rules = _generate_iree_rules(common_rule_factory, iree_artifacts_dir,
                                    iree_module_generation_configs)
  # Currently the rules are simple so the common rules can be always put at the
  # top. Need a topological sort once the dependency gets complicated.
  return common_rule_factory.generate_cmake_rules() + iree_rules
