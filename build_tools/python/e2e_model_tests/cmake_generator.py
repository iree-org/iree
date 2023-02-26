## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake rules for e2e model tests."""

from typing import List

from e2e_model_tests import test_definitions, run_module_utils
from e2e_test_artifacts import iree_artifacts
from e2e_test_framework.definitions import iree_definitions
import cmake_builder.rules


def generate_rules(
    module_generation_configs: List[iree_definitions.ModuleGenerationConfig]
) -> List[str]:
  """Generates CMake rules for e2e model tests."""

  gen_config_lookup_map = {}
  for gen_config in module_generation_configs:
    gen_config_lookup_map[(gen_config.imported_model.composite_id(),
                           gen_config.compile_config.id)] = gen_config

  cmake_rules = []
  for test_config in test_definitions.TEST_CONFIGS:
    for platform in test_definitions.CMakePlatform:
      if platform in test_config.unsupported_platforms:
        continue

      imported_model_id = test_config.imported_model.composite_id()
      compile_config_id = test_definitions.PLATFORM_COMPILE_CONFIG_MAP[
          platform].id
      gen_config = gen_config_lookup_map[(imported_model_id, compile_config_id)]

      module_path = iree_artifacts.get_module_dir_path(
          gen_config) / iree_artifacts.MODULE_FILENAME
      cmake_rule = cmake_builder.rules.build_set(
          variable_name=f"IREE_TEST_MODULE_{imported_model_id}_{platform.value}"
          .upper(),
          value=f'"{module_path}"')
      cmake_rules.append(cmake_rule)

  for test_config in test_definitions.TEST_CONFIGS:
    imported_model = test_config.imported_model
    runner_args = run_module_utils.build_run_flags_for_model(
        model=imported_model.model,
        model_input_data=test_config.input_data) + test_config.extra_test_flags
    # TODO(#11136): Currently the DRIVER is a separate field in the CMake rule (
    # and has effect on test labels). Rules should be generated in another way
    # to avoid that. Generates the flags without the driver for now.
    runner_args += run_module_utils.build_run_flags_for_execution_config(
        test_config.execution_config, with_driver=False)
    cmake_rule = cmake_builder.rules.build_iree_benchmark_suite_module_test(
        target_name=test_config.name,
        imported_model=imported_model.composite_id(),
        driver=test_config.execution_config.driver.value,
        expected_output=test_config.expected_output,
        runner_args=runner_args,
        xfail_platforms=[
            platform.value for platform in test_config.xfail_platforms
        ],
        unsupported_platforms=[
            platform.value for platform in test_config.unsupported_platforms
        ])
    cmake_rules.append(cmake_rule)

  return cmake_rules
