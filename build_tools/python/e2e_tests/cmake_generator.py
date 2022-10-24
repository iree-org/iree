## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake rules for e2e model tests."""

import pathlib
from typing import List

from e2e_tests import benchmark_module_tests, e2e_tests_definitions
from e2e_test_framework.definitions import common_definitions
import e2e_test_artifacts.artifacts
import cmake_builder.rules


def __get_iree_run_module_args(
    test_config: e2e_tests_definitions.E2EModuleTestConfig) -> List[str]:
  model = test_config.module_generation_config.imported_model.model
  args = [f"--entry_function={model.entry_function}"]
  if test_config.input_data.data_format != common_definitions.InputDataFormat.ZERO:
    raise ValueError("Currently only support all-zero data.")
  args += [
      f"--function_input={input_type}=0" for input_type in model.input_types
  ]
  args += test_config.extra_test_flags
  return args


def generate_rules(root_path: pathlib.PurePath) -> List[str]:
  test_configs = benchmark_module_tests.generate_tests()
  artifacts_root = e2e_test_artifacts.artifacts.generate_default_artifacts_root(
  )

  cmake_rules = []
  for test_config in test_configs:
    generation_config = test_config.module_generation_config
    model = generation_config.imported_model.model
    execution_config = test_config.module_execution_config
    iree_model_dir = artifacts_root.iree_artifacts_root.model_dir_map[model.id]
    iree_module_dir = iree_model_dir.module_dir_map[
        generation_config.compile_config.id]
    module_src_path = str(root_path / iree_module_dir.module_path)
    cmake_rule = cmake_builder.rules.build_iree_run_module_test(
        target_name=test_config.name,
        module_src=module_src_path,
        driver=execution_config.driver.value,
        expected_output=test_config.expected_output,
        runner_args=__get_iree_run_module_args(test_config))
    cmake_rules.append(cmake_rule)

  return cmake_rules
