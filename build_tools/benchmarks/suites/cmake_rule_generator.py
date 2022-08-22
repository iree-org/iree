## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generator that generates CMake rules from python defined benchmarks.

The rules will build required artifacts to run benchmarks.
"""

from dataclasses import dataclass
from typing import List, Optional
import os
import pathlib
import string
import urllib.parse

from .definitions import common_definitions, iree_definitions
from . import iree_benchmarks

# Template dir: build_tools/benchmarks/suites/../../cmake
TEMPLATE_DIR = pathlib.Path(__file__).parents[2] / "cmake"
DOWNLOAD_ARTIFACT_CMAKE_TEMPLATE = string.Template(
    open(TEMPLATE_DIR / "iree_download_artifact_template.cmake", "r").read())
TFLITE_IMPORT_CMAKE_TEMPLATE = string.Template(
    open(TEMPLATE_DIR / "iree_tflite_import_template.cmake", "r").read())
TF_IMPORT_CMAKE_TEMPLATE = string.Template(
    open(TEMPLATE_DIR / "iree_tf_import_template.cmake", "r").read())


@dataclass
class ModelRule(object):
  target_name: str
  file_path: str
  cmake_rule: str


@dataclass
class IreeModelImportRule(object):
  target_name: str
  output_file_path: str
  mlir_dialect_type: str
  cmake_rule: Optional[str]


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
    _, file_ext = os.path.splitext(model_url.path)
    # Model path: <model_artifacts_dir>/<model_id>_<model_name>.<file ext>
    model_path = f"{self._model_artifacts_dir}/{model.id}_{model.name}{file_ext}"

    if model_url.scheme == "https":
      cmake_rule = DOWNLOAD_ARTIFACT_CMAKE_TEMPLATE.substitute(
          _TARGET_NAME_=target_name,
          _OUTPUT_PATH_=model_path,
          _SOURCE_URL_=model.source_url)
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
    self._generate_flagfile_rules = {}

  def add_import_model_rule(
      self,
      model_id: str,
      model_name: str,
      model_source_type: common_definitions.ModelSourceType,
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
          _TARGET_NAME_=target_name,
          _SOURCE_MODEL_PATH_=source_model_rule.file_path,
          _OUTPUT_PATH_=output_file_path)
      mlir_dialect_type = "tosa"
    elif model_source_type == common_definitions.ModelSourceType.EXPORTED_TF:
      cmake_rule = TF_IMPORT_CMAKE_TEMPLATE.substitute(
          _TARGET_NAME_=target_name,
          _SOURCE_MODEL_PATH_=source_model_rule.file_path,
          _OUTPUT_PATH_=output_file_path)
      mlir_dialect_type = "mhlo"
    else:
      raise ValueError(
          f"Unsupported source type '{model_source_type}' of the model '{model_id}'."
      )

    import_model_rule = IreeModelImportRule(target_name=target_name,
                                            output_file_path=output_file_path,
                                            mlir_dialect_type=mlir_dialect_type,
                                            cmake_rule=cmake_rule)

    self._import_model_rules[model_id] = import_model_rule
    return import_model_rule

  def generate_cmake_rules(self) -> List[str]:
    """Dump all cmake rules in a correct order."""
    import_model_rules = [
        rule.cmake_rule for rule in self._import_model_rules.values()
    ]
    return import_model_rules


def _generate_iree_benchmark_rules(common_rule_factory: CommonRuleFactory,
                                   iree_artifacts_dir: str) -> List[str]:
  iree_rule_factory = IreeRuleFactory(iree_artifacts_dir)
  compile_specs, _ = iree_benchmarks.Linux_x86_64_Benchmarks.generate()
  for compile_spec in compile_specs:
    model = compile_spec.model

    source_model_rule = common_rule_factory.add_model_rule(model)
    iree_rule_factory.add_import_model_rule(model_id=model.id,
                                            model_name=model.name,
                                            model_source_type=model.source_type,
                                            source_model_rule=source_model_rule)

    # TODO(pzread): Generate the compilation and run rules.

  return iree_rule_factory.generate_cmake_rules()


def generate_benchmark_rules(model_artifacts_dir: str,
                             iree_artifacts_dir: str) -> List[str]:
  """Generates cmake rules for all benchmarks.
  
  Args:
    model_artifacts_dir: root directory to store model files. Can contain CMake
      variable syntax in the path.
    iree_artifacts_dir: root directory to store generated IREE artifacts. Can
      contain CMake variable syntax in the path.
  Returns:
    List of CMake rules.
  """
  common_rule_factory = CommonRuleFactory(model_artifacts_dir)
  iree_rules = _generate_iree_benchmark_rules(common_rule_factory,
                                              iree_artifacts_dir)
  # Currently the rules are simple so the common rules can be always put at the
  # top. Need a topological sort once the dependency gets complicated.
  return common_rule_factory.generate_cmake_rules() + iree_rules
