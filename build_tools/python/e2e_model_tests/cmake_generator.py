## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake rules for e2e model tests."""

from typing import List

from e2e_test_framework.definitions import common_definitions
from e2e_model_tests import test_definitions
import cmake_builder.rules


def __get_iree_run_module_args(
    test_config: test_definitions.ModelTestConfig) -> List[str]:
  model = test_config.model
  args = [f"--entry_function={model.entry_function}"]
  if test_config.input_data != common_definitions.ZEROS_MODEL_INPUT_DATA:
    raise ValueError("Currently only support all-zeros data.")
  args += [
      f"--function_input={input_type}=0" for input_type in model.input_types
  ]
  args += test_config.extra_test_flags
  return args


def generate_rules() -> List[str]:
  """Generates CMake rules for e2e model tests."""
  cmake_rules = []

  for platform in test_definitions.CMakePlatform:
    cmake_rule = cmake_builder.rules.build_set(
        variable_name=f"IREE_MODULE_COMPILE_CONFIG_ID_{platform.value.upper()}",
        value=f'"{test_definitions.PLATFORM_COMPILE_CONFIG_MAP[platform].id}"')
    cmake_rules.append(cmake_rule)

  for test_config in test_definitions.TEST_CONFIGS:
    model = test_config.model
    execution_config = test_config.execution_config
    cmake_rule = cmake_builder.rules.build_iree_benchmark_suite_module_test(
        target_name=test_config.name,
        model=f"{model.id}_{model.name}",
        driver=execution_config.driver.value,
        expected_output=test_config.expected_output,
        runner_args=__get_iree_run_module_args(test_config),
        xfail_platforms=[
            platform.value for platform in test_config.xfail_platforms
        ],
        unsupported_platforms=[
            platform.value for platform in test_config.unsupported_platforms
        ])
    cmake_rules.append(cmake_rule)

  return cmake_rules
