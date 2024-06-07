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

from benchmark_suites.iree import benchmark_presets
from e2e_test_artifacts import iree_artifacts, utils
from e2e_test_artifacts.cmake_generator import model_rule_generator
from e2e_test_framework.definitions import iree_definitions
import cmake_builder.rules

# Imported models for default benchmarks.
BENCHMARK_IMPORT_MODELS_CMAKE_TARGET = "iree-benchmark-import-models"
# Imported models for large benchmarks.
LARGE_BENCHMARK_IMPORT_MODELS_CMAKE_TARGET = "iree-benchmark-import-models-large"
# Prefix of benchmark suite cmake targets.
BENCHMARK_SUITES_CMAKE_TARGET_PREFIX = "iree-benchmark-suites-"


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
        self,
        source_model_rule: model_rule_generator.ModelRule,
        imported_model: iree_definitions.ImportedModel,
        output_file_path: pathlib.PurePath,
    ) -> IreeModelImportRule:
        model = imported_model.model
        import_config = imported_model.import_config
        if import_config.tool == iree_definitions.ImportTool.NONE:
            if source_model_rule.file_path != output_file_path:
                raise ValueError(
                    f"Separate path for MLIR model isn't supported yet: "
                    f"('{source_model_rule.file_path }' != '{output_file_path}')"
                )
            return IreeModelImportRule(
                target_name=source_model_rule.target_name,
                output_file_path=output_file_path,
                cmake_rules=[],
            )

        # Import target name: iree-imported-model-<imported_model_name>
        target_name = utils.get_safe_name(f"iree-imported-model-{imported_model.name}")

        import_flags = import_config.materialize_import_flags(model)
        if import_config.tool == iree_definitions.ImportTool.TFLITE_IMPORTER:
            cmake_rules = [
                cmake_builder.rules.build_iree_import_tflite_model(
                    target_path=self.build_target_path(target_name),
                    source=str(source_model_rule.file_path.as_posix()),
                    import_flags=import_flags,
                    output_mlir_file=str(output_file_path.as_posix()),
                )
            ]
        elif import_config.tool == iree_definitions.ImportTool.TF_IMPORTER:
            cmake_rules = [
                cmake_builder.rules.build_iree_import_tf_model(
                    target_path=self.build_target_path(target_name),
                    source=str(source_model_rule.file_path.as_posix()),
                    import_flags=import_flags,
                    output_mlir_file=str(output_file_path.as_posix()),
                )
            ]
        else:
            raise ValueError(
                f"Unsupported import tool '{import_config.tool}' of the model '{model.id}'."
            )

        return IreeModelImportRule(
            target_name=target_name,
            output_file_path=output_file_path,
            cmake_rules=cmake_rules,
        )

    def build_module_compile_rule(
        self,
        model_import_rule: IreeModelImportRule,
        module_generation_config: iree_definitions.ModuleGenerationConfig,
        output_file_path: pathlib.PurePath,
    ) -> IreeModuleCompileRule:
        compile_flags = module_generation_config.materialize_compile_flags(
            module_dir_path=output_file_path.parent
        )

        # Module target name: iree-module-<gen_config_name>
        target_name = utils.get_safe_name(
            f"iree-module-{module_generation_config.name}"
        )

        cmake_rules = [
            cmake_builder.rules.build_iree_bytecode_module(
                target_name=target_name,
                src=str(model_import_rule.output_file_path.as_posix()),
                module_name=str(output_file_path.as_posix()),
                flags=compile_flags,
                friendly_name=str(module_generation_config),
            )
        ]

        # TODO(#10155): Dump the compile flags from iree_bytecode_module into a flagfile.

        return IreeModuleCompileRule(
            target_name=target_name,
            output_module_path=output_file_path,
            cmake_rules=cmake_rules,
        )

    def build_target_path(self, target_name: str):
        """Returns the full target path by combining the package name and the target
        name.
        """
        return f"{self._package_name}_{target_name}"


def generate_rules(
    package_name: str,
    root_path: pathlib.PurePath,
    gen_configs: Sequence[iree_definitions.ModuleGenerationConfig],
    run_configs: Sequence[iree_definitions.E2EModelRunConfig],
    model_rule_map: Dict[str, model_rule_generator.ModelRule],
) -> List[str]:
    """Generates all rules to build IREE artifacts.

    Args:
      package_name: CMake package name for rules.
      root_path: path of the root artifact directory.
      gen_configs: full list of IREE module generation configs of both
        compilation and execution benchmarks.
      run_configs: full list of IREE E2E model run configs to calculate the
        artifact dependencies of exectuion benchmarks.
      model_rule_map: map of generated model rules keyed by model id, it must
        cover all model referenced in gen_configs.
    Returns:
      List of cmake rules.
    """

    rule_builder = IreeRuleBuilder(package_name=package_name)

    all_imported_models = dict(
        (config.imported_model.composite_id, config.imported_model)
        for config in gen_configs
    )

    cmake_rules = []
    model_import_rule_map = {}
    for imported_model_id, imported_model in all_imported_models.items():
        model_rule = model_rule_map.get(imported_model.model.id)
        if model_rule is None:
            raise ValueError(f"Model rule not found for {imported_model.model.id}.")

        imported_model_path = iree_artifacts.get_imported_model_path(
            imported_model=imported_model, root_path=root_path
        )
        model_import_rule = rule_builder.build_model_import_rule(
            source_model_rule=model_rule,
            imported_model=imported_model,
            output_file_path=imported_model_path,
        )
        model_import_rule_map[imported_model_id] = model_import_rule
        cmake_rules.extend(model_import_rule.cmake_rules)

    gen_config_to_presets = collections.defaultdict(set)
    for gen_config in gen_configs:
        gen_config_to_presets[gen_config.composite_id].update(gen_config.presets)
    # Include the presets from dependent run configs, so they are built for
    # execution benchmark presets.
    for run_config in run_configs:
        gen_config = run_config.module_generation_config
        gen_config_to_presets[gen_config.composite_id].update(run_config.presets)

    suites_to_deps = collections.defaultdict(set)
    for gen_config in gen_configs:
        model_import_rule = model_import_rule_map[
            gen_config.imported_model.composite_id
        ]
        module_dir_path = iree_artifacts.get_module_dir_path(
            module_generation_config=gen_config, root_path=root_path
        )
        module_compile_rule = rule_builder.build_module_compile_rule(
            model_import_rule=model_import_rule,
            module_generation_config=gen_config,
            output_file_path=module_dir_path / iree_artifacts.MODULE_FILENAME,
        )
        cmake_rules.extend(module_compile_rule.cmake_rules)

        presets = gen_config_to_presets[gen_config.composite_id]
        # A benchmark can be in default and large presets at the same time. For
        # example, batch-1 benchmark is added to default for sanity check. So
        # check both cases.
        if presets.intersection(benchmark_presets.DEFAULT_PRESETS):
            presets.add("default")
        if presets.intersection(benchmark_presets.LARGE_PRESETS):
            presets.add("large")

        for preset in presets:
            preset_target = f"{BENCHMARK_SUITES_CMAKE_TARGET_PREFIX}{preset}"
            suites_to_deps[preset_target].add(module_compile_rule.target_name)

        import_target = model_import_rule.target_name
        if "default" in presets:
            suites_to_deps[BENCHMARK_IMPORT_MODELS_CMAKE_TARGET].add(import_target)
        if "large" in presets:
            suites_to_deps[LARGE_BENCHMARK_IMPORT_MODELS_CMAKE_TARGET].add(
                import_target
            )

    # The dict suites_to_deps is inserted in a quite arbitrary order. Sort it by
    # the key first.
    for suite_target in sorted(suites_to_deps.keys()):
        target_names = suites_to_deps[suite_target]
        cmake_rules.append(
            cmake_builder.rules.build_add_dependencies(
                target=suite_target,
                deps=[
                    rule_builder.build_target_path(target_name)
                    for target_name in sorted(target_names)
                ],
            )
        )

    return cmake_rules
