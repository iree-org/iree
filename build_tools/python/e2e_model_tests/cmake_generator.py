## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake rules for e2e model tests."""

import pathlib
from typing import List
from build_tools.python.e2e_model_tests import benchmark_model_tests

from e2e_model_tests import test_definitions
from e2e_test_framework.definitions import common_definitions, iree_definitions
import e2e_test_artifacts.artifacts
import cmake_builder.rules


def __get_iree_run_module_args(
    test_config: test_definitions.ModelTestConfig) -> List[str]:
  model = test_config.module_generation_config.imported_model.model
  args = [f"--entry_function={model.entry_function}"]
  if test_config.input_data.data_format != common_definitions.InputDataFormat.ZERO:
    raise ValueError("Currently only support all-zeros data.")
  args += [
      f"--function_input={input_type}=0" for input_type in model.input_types
  ]
  args += test_config.extra_test_flags
  return args


ABI_TO_CMAKE_PLATFORM_MAP = {
    iree_definitions.TargetABI.LINUX_ANDROID29: "Android",
    iree_definitions.TargetABI.LINUX_ANDROID31: "Android",
    iree_definitions.TargetABI.LINUX_GNU: "Linux",
}
ARCH_TO_CMAKE_SYSTEM_PROCESSOR_MAP = {
    common_definitions.DeviceArchitecture.RV32_GENERIC: "riscv32",
    common_definitions.DeviceArchitecture.RV64_GENERIC: "riscv64",
    common_definitions.DeviceArchitecture.X86_64_CASCADELAKE: "x86_64",
    common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC: "arm64-v8a",
}


def generate_rules(root_path: pathlib.PurePath) -> List[str]:
  test_configs = benchmark_model_tests.generate_tests()
  artifacts_root = (
      e2e_test_artifacts.artifacts.generate_default_artifacts_root())

  cmake_rules = []
  for test_config in test_configs:
    generation_config = test_config.module_generation_config
    execution_config = test_config.module_execution_config
    model = generation_config.imported_model.model

    compile_config = generation_config.compile_config
    supported_platforms = []
    for target in compile_config.compile_targets:
      platform = ABI_TO_CMAKE_PLATFORM_MAP[target.target_abi]
      system_processor = ARCH_TO_CMAKE_SYSTEM_PROCESSOR_MAP[
          target.target_architecture]
      supported_platforms.append(f"{system_processor}-{platform}")

    iree_model_dir = artifacts_root.iree_artifacts_root.model_dir_map[model.id]
    iree_module_dir = iree_model_dir.module_dir_map[
        generation_config.compile_config.id]
    module_src_path = str(root_path / iree_module_dir.module_path)

    cmake_rule = cmake_builder.rules.build_iree_run_module_test(
        target_name=test_config.name,
        module_src=module_src_path,
        driver=execution_config.driver.value,
        expected_output=test_config.expected_output,
        runner_args=__get_iree_run_module_args(test_config),
        supported_platforms=supported_platforms)
    cmake_rules.append(cmake_rule)

  return cmake_rules
