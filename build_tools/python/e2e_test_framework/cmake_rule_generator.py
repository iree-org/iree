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
import string
import urllib.parse

from e2e_test_framework.definitions import common_definitions, iree_definitions


def read_template_from_file(template_path: pathlib.Path) -> string.Template:
  return string.Template(template_path.read_text())


TEMPLATE_DIR = pathlib.Path(__file__).parent
DOWNLOAD_ARTIFACT_CMAKE_TEMPLATE = read_template_from_file(
    TEMPLATE_DIR / "iree_download_artifact_template.cmake")
TFLITE_IMPORT_CMAKE_TEMPLATE = read_template_from_file(
    TEMPLATE_DIR / "iree_tflite_import_template.cmake")
TF_IMPORT_CMAKE_TEMPLATE = read_template_from_file(
    TEMPLATE_DIR / "iree_tf_import_template.cmake")
IREE_BYTECODE_MODULE_CMAKE_TEMPLATE = read_template_from_file(
    TEMPLATE_DIR / "iree_bytecode_module_template.cmake")

# Archive extensions used to pack models.
ARCHIVE_FILE_EXTENSIONS = [".tar", ".gz"]


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
      cmake_rule = DOWNLOAD_ARTIFACT_CMAKE_TEMPLATE.substitute(
          __TARGET_NAME=target_name,
          __OUTPUT_PATH=model_path,
          __SOURCE_URL=model.source_url)
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
      model_id: str,
      model_name: str,
      model_source_type: common_definitions.ModelSourceType,
      model_entry_function: str,
      source_model_rule: ModelRule,
  ) -> IreeModelImportRule:
    """Adds a rule to fetch the model and import into MLIR. Reuses the rule when
    possible."""

    if model_id in self._import_model_rules:
      return self._import_model_rules[model_id]

    # If the source model is MLIR, no import rule is needed.
    if model_source_type == common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR:
      import_model_rule = IreeModelImportRule(
          target_name=source_model_rule.target_name,
          model_id=model_id,
          model_name=model_name,
          output_file_path=source_model_rule.file_path,
          mlir_dialect_type="linalg",
          cmake_rule=None)
      self._import_model_rules[model_id] = import_model_rule
      return import_model_rule

    # Import target: <package_name>_iree-import-model-<model_id>
    target_name = f"iree-import-model-{model_id}"

    # Imported MLIR path: <iree_artifacts_dir>/<model_id>_<model_name>/<model_name>.mlir
    output_file_path = f"{self._iree_artifacts_dir}/{model_id}_{model_name}/{model_name}.mlir"

    if model_source_type == common_definitions.ModelSourceType.EXPORTED_TFLITE:
      cmake_rule = TFLITE_IMPORT_CMAKE_TEMPLATE.substitute(
          __TARGET_NAME=target_name,
          __SOURCE_MODEL_PATH=source_model_rule.file_path,
          __OUTPUT_PATH=output_file_path)
      mlir_dialect_type = "tosa"
    elif model_source_type == common_definitions.ModelSourceType.EXPORTED_TF:
      cmake_rule = TF_IMPORT_CMAKE_TEMPLATE.substitute(
          __TARGET_NAME=target_name,
          __SOURCE_MODEL_PATH=source_model_rule.file_path,
          __ENTRY_FUNCTION=model_entry_function,
          __OUTPUT_PATH=output_file_path)
      mlir_dialect_type = "mhlo"
    else:
      raise ValueError(
          f"Unsupported source type '{model_source_type}' of the model '{model_id}'."
      )

    import_model_rule = IreeModelImportRule(target_name=target_name,
                                            model_id=model_id,
                                            model_name=model_name,
                                            output_file_path=output_file_path,
                                            mlir_dialect_type=mlir_dialect_type,
                                            cmake_rule=cmake_rule)

    self._import_model_rules[model_id] = import_model_rule
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

    cmake_rule = IREE_BYTECODE_MODULE_CMAKE_TEMPLATE.substitute(
        __TARGET_NAME=target_name,
        __MODULE_OUTPUT_PATH=output_path,
        __SOURCE_MODEL_PATH=model_import_rule.output_file_path,
        __COMPILE_FLAGS=";".join(compile_flags),
        __SOURCE_MODEL_TARGET=model_import_rule.target_name)
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
          f"--iree-llvm-target-triple=x86_64-unknown-{target.target_platform.value}",
          f"--iree-llvm-target-cpu={arch_info.microarchitecture.lower()}"
      ]
    else:
      raise ValueError(f"Unsupported architecture '{arch_info.architecture}'")
    return flags


def _generate_iree_rules(
    common_rule_factory: CommonRuleFactory, iree_artifacts_dir: str,
    model_compile_configs: Sequence[iree_definitions.ModelCompileConfig]
) -> List[str]:
  iree_rule_factory = IreeRuleFactory(iree_artifacts_dir)
  for model_compile_config in model_compile_configs:
    model = model_compile_config.model
    compile_config = model_compile_config.compile_config

    source_model_rule = common_rule_factory.add_model_rule(model)
    import_rule = iree_rule_factory.add_import_model_rule(
        model_id=model.id,
        model_name=model.name,
        model_source_type=model.source_type,
        model_entry_function=model.entry_function,
        source_model_rule=source_model_rule)
    iree_rule_factory.add_compile_module_rule(compile_config=compile_config,
                                              model_import_rule=import_rule)

  return iree_rule_factory.generate_cmake_rules()


def generate_rules(
    model_artifacts_dir: str, iree_artifacts_dir: str,
    iree_model_compile_configs: Sequence[iree_definitions.ModelCompileConfig]
) -> List[str]:
  """Generates cmake rules to build benchmarks.
  
  Args:
    model_artifacts_dir: root directory to store model files. Can contain CMake
      variable syntax in the path.
    iree_artifacts_dir: root directory to store generated IREE artifacts. Can
      contain CMake variable syntax in the path.
    iree_model_compile_configs: compile configs for IREE targets.
  Returns:
    List of CMake rules.
  """
  common_rule_factory = CommonRuleFactory(model_artifacts_dir)
  iree_rules = _generate_iree_rules(common_rule_factory, iree_artifacts_dir,
                                    iree_model_compile_configs)
  # Currently the rules are simple so the common rules can be always put at the
  # top. Need a topological sort once the dependency gets complicated.
  return common_rule_factory.generate_cmake_rules() + iree_rules
