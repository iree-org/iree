## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake rules for e2e model tests."""

from typing import List

from e2e_model_tests import test_definitions, run_module_utils
import cmake_builder.rules


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
    runner_args = run_module_utils.build_run_flags_for_model(
        model=model,
        model_input_data=test_config.input_data) + test_config.extra_test_flags
    # TODO(#11136): Currently the DRIVER is a separate field in the CMake rule (
    # and has effect on test labels). Generates the flags without the driver.
    runner_args += run_module_utils.build_run_flags_for_execution_config(
        test_config.execution_config, without_driver=True)
    cmake_rule = cmake_builder.rules.build_iree_benchmark_suite_module_test(
        target_name=test_config.name,
        model=f"{model.id}_{model.name}",
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
