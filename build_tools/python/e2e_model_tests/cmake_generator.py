## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake rules for e2e model tests."""

from typing import List

from e2e_model_tests import test_definitions
from e2e_test_artifacts import iree_artifacts
from e2e_test_framework.definitions import iree_definitions
import cmake_builder.rules


def generate_rules(
    module_generation_configs: List[iree_definitions.ModuleGenerationConfig]
) -> List[str]:
  """Generates CMake rules for e2e model tests."""

  # ModelTestConfig uses (imported_model, compile_config (mapped from platform))
  # to define the required module. Collect module paths indexed by the pair.
  all_module_path_map = {}
  for gen_config in module_generation_configs:
    module_path = iree_artifacts.get_module_dir_path(
        gen_config) / iree_artifacts.MODULE_FILENAME
    all_module_path_map[(gen_config.imported_model.composite_id,
                         gen_config.compile_config.id)] = module_path

  cmake_rules = []
  for test_config in test_definitions.TEST_CONFIGS:
    imported_model = test_config.imported_model
    platform_module_map = {}
    for platform in test_definitions.CMakePlatform:
      if platform in test_config.unsupported_platforms:
        continue

      compile_config = test_definitions.PLATFORM_COMPILE_CONFIG_MAP[platform]
      module_path = all_module_path_map.get(
          (imported_model.composite_id, compile_config.id))
      if module_path is None:
        raise ValueError(
            f"Module for {test_config.name} on {platform} not found.")
      platform_module_map[platform.value] = module_path

    # TODO(#11136): Currently the DRIVER is a separate field in the CMake rule (
    # and has effect on test labels). Rules should be generated in another way
    # to avoid that. Generates the flags without the driver for now.
    runner_args = iree_definitions.generate_run_flags(
        imported_model=imported_model,
        input_data=test_config.input_data,
        module_execution_config=test_config.execution_config,
        with_driver=False) + test_config.extra_test_flags
    cmake_rule = cmake_builder.rules.build_iree_benchmark_suite_module_test(
        target_name=test_config.name,
        driver=test_config.execution_config.driver.value,
        expected_output=test_config.expected_output,
        platform_module_map=platform_module_map,
        runner_args=runner_args,
        xfail_platforms=[
            platform.value for platform in test_config.xfail_platforms
        ])
    cmake_rules.append(cmake_rule)

  return cmake_rules
