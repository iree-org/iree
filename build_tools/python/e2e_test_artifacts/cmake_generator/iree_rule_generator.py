## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake rules to build IREE artifacts."""

import collections
from dataclasses import dataclass
from typing import Dict, List, Sequence
import pathlib

from benchmark_suites.iree import benchmark_tags
from e2e_test_artifacts import iree_artifacts
from e2e_test_artifacts.cmake_generator import model_rule_generator
from e2e_test_framework.definitions import iree_definitions
import cmake_builder.rules

BENCHMARK_IMPORT_MODELS_CMAKE_TARGET = "iree-benchmark-import-models"
# Default benchmark suites.
BENCHMARK_SUITES_CMAKE_TARGET = "iree-benchmark-suites"
# Compilation statistics suites for default benchmarks.
E2E_COMPILE_STATS_SUITES = "iree-e2e-compile-stats-suites"
# Large benchmark suites.
LARGE_BENCHMARK_SUITES_CMAKE_TARGET = "iree-benchmark-suites-large"
# Compilation statistics suites for large benchmarks.
LARGE_E2E_COMPILE_STATS_SUITES_CMAKE_TARGET = "iree-e2e-compile-stats-suites-large"


@dataclass(frozen=True)
class IreeModelImportRule(object):
  target_name: str
  output_file_path: pathlib.PurePath
  cmake_rules: List[str]


@dataclass(frozen=True)
class IreeModuleCompileRule(object):
  target_name: str
  output_module_path: pathlib.PurePath
  cmake_rules: List[str]


class IreeRuleBuilder(object):
  """Builder to generate IREE CMake rules."""

  _package_name: str

  def __init__(self, package_name: str):
    self._package_name = package_name

  def build_model_import_rule(
      self, source_model_rule: model_rule_generator.ModelRule,
      imported_model: iree_definitions.ImportedModel,
      output_file_path: pathlib.PurePath) -> IreeModelImportRule:

    model = imported_model.model
    import_config = imported_model.import_config
    if import_config.tool == iree_definitions.ImportTool.NONE:
      if source_model_rule.file_path != output_file_path:
        raise ValueError(
            f"Separate path for MLIR model isn't supported yet: "
            f"('{source_model_rule.file_path }' != '{output_file_path}')")
      return IreeModelImportRule(target_name=source_model_rule.target_name,
                                 output_file_path=output_file_path,
                                 cmake_rules=[])

    # Import target name: iree-imported-model-<imported_model_id>
    target_name = f"iree-imported-model-{imported_model.composite_id}"

    import_flags = import_config.materialize_import_flags(model)
    if import_config.tool == iree_definitions.ImportTool.TFLITE_IMPORTER:
      cmake_rules = [
          cmake_builder.rules.build_iree_import_tflite_model(
              target_path=self.build_target_path(target_name),
              source=str(source_model_rule.file_path),
              import_flags=import_flags,
              output_mlir_file=str(output_file_path))
      ]
    elif import_config.tool == iree_definitions.ImportTool.TF_IMPORTER:
      cmake_rules = [
          cmake_builder.rules.build_iree_import_tf_model(
              target_path=self.build_target_path(target_name),
              source=str(source_model_rule.file_path),
              import_flags=import_flags,
              output_mlir_file=str(output_file_path))
      ]
    else:
      raise ValueError(
          f"Unsupported import tool '{import_config.tool}' of the model '{model.id}'."
      )

    return IreeModelImportRule(target_name=target_name,
                               output_file_path=output_file_path,
                               cmake_rules=cmake_rules)

  def build_module_compile_rule(
      self, model_import_rule: IreeModelImportRule,
      module_generation_config: iree_definitions.ModuleGenerationConfig,
      output_file_path: pathlib.PurePath) -> IreeModuleCompileRule:

    compile_flags = module_generation_config.materialize_compile_flags(
        module_dir_path=output_file_path.parent)

    # Module target name: iree-module-<gen_config_id>
    target_name = f"iree-module-{module_generation_config.composite_id}"

    cmake_rules = [
        cmake_builder.rules.build_iree_bytecode_module(
            target_name=target_name,
            src=str(model_import_rule.output_file_path),
            module_name=str(output_file_path),
            flags=compile_flags,
            friendly_name=str(module_generation_config))
    ]

    # TODO(#10155): Dump the compile flags from iree_bytecode_module into a flagfile.

    return IreeModuleCompileRule(target_name=target_name,
                                 output_module_path=output_file_path,
                                 cmake_rules=cmake_rules)

  def build_target_path(self, target_name: str):
    """Returns the full target path by combining the package name and the target
    name.
    """
    return f"{self._package_name}_{target_name}"


def generate_rules(
    package_name: str, root_path: pathlib.PurePath,
    module_generation_configs: Sequence[
        iree_definitions.ModuleGenerationConfig],
    model_rule_map: Dict[str, model_rule_generator.ModelRule]) -> List[str]:
  """Generates all rules to build IREE artifacts.

  Args:
    package_name: CMake package name for rules.
    root_path: path of the root artifact directory.
    module_generation_configs: list of IREE module generation configs.
    model_rule_map: map of generated model rules keyed by model id, it must
      cover all model referenced in module_generation_configs.
  Returns:
    List of cmake rules.
  """

  rule_builder = IreeRuleBuilder(package_name=package_name)

  all_imported_models = dict(
      (config.imported_model.composite_id, config.imported_model)
      for config in module_generation_configs)

  cmake_rules = []
  model_import_rule_map = {}
  for imported_model_id, imported_model in all_imported_models.items():
    model_rule = model_rule_map.get(imported_model.model.id)
    if model_rule is None:
      raise ValueError(f"Model rule not found for {imported_model.model.id}.")

    imported_model_path = iree_artifacts.get_imported_model_path(
        imported_model=imported_model, root_path=root_path)
    model_import_rule = rule_builder.build_model_import_rule(
        source_model_rule=model_rule,
        imported_model=imported_model,
        output_file_path=imported_model_path)
    model_import_rule_map[imported_model_id] = model_import_rule
    cmake_rules.extend(model_import_rule.cmake_rules)

  suite_target_names = collections.defaultdict(list)
  for gen_config in module_generation_configs:
    model_import_rule = model_import_rule_map[
        gen_config.imported_model.composite_id]
    module_dir_path = iree_artifacts.get_module_dir_path(
        module_generation_config=gen_config, root_path=root_path)
    module_compile_rule = rule_builder.build_module_compile_rule(
        model_import_rule=model_import_rule,
        module_generation_config=gen_config,
        output_file_path=module_dir_path / iree_artifacts.MODULE_FILENAME)

    is_compile_stats = (benchmark_tags.COMPILE_STATS
                        in gen_config.compile_config.tags)
    if benchmark_tags.LARGE in gen_config.tags:
      if is_compile_stats:
        suite_target = LARGE_E2E_COMPILE_STATS_SUITES_CMAKE_TARGET
      else:
        suite_target = LARGE_BENCHMARK_SUITES_CMAKE_TARGET
    else:
      if is_compile_stats:
        suite_target = E2E_COMPILE_STATS_SUITES
      else:
        suite_target = BENCHMARK_SUITES_CMAKE_TARGET

    suite_target_names[suite_target].append(module_compile_rule.target_name)

    cmake_rules.extend(module_compile_rule.cmake_rules)

  if len(model_import_rule_map) > 0:
    cmake_rules.append(
        cmake_builder.rules.build_add_dependencies(
            target=BENCHMARK_IMPORT_MODELS_CMAKE_TARGET,
            deps=[
                rule_builder.build_target_path(rule.target_name)
                for rule in model_import_rule_map.values()
            ]))

  for suite_target, module_target_names in suite_target_names.items():
    cmake_rules.append(
        cmake_builder.rules.build_add_dependencies(
            target=suite_target,
            deps=[
                rule_builder.build_target_path(target_name)
                for target_name in module_target_names
            ]))

  return cmake_rules
