## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines e2e model tests on benchmark models."""

from dataclasses import dataclass
from typing import List
import dataclasses
import enum

from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.definitions import iree_definitions
from e2e_test_framework.models import tflite_models
from benchmark_suites.iree import (riscv_benchmarks, x86_64_benchmarks,
                                   armv8_a_benchmarks, module_execution_configs)
import cmake_builder.rules


class CMakePlatform(enum.Enum):
  """Enum of CMake system platform string."""
  ANDROID_ARMV8_A = "android-arm64-v8a"
  LINUX_RISCV32 = "riscv32-linux"
  LINUX_RISCV64 = "riscv64-linux"
  LINUX_X86_64 = "x86_64-linux"


# Compile config used for each CMake system platform.
PLATFORM_COMPILE_CONFIG_MAP = {
    CMakePlatform.ANDROID_ARMV8_A:
        armv8_a_benchmarks.Android_ARMv8_A_Benchmarks.DEFAULT_COMPILE_CONFIG,
    CMakePlatform.LINUX_RISCV32:
        riscv_benchmarks.Linux_RV32_Benchmarks.DEFAULT_COMPILE_CONFIG,
    CMakePlatform.LINUX_RISCV64:
        riscv_benchmarks.Linux_RV64_Benchmarks.DEFAULT_COMPILE_CONFIG,
    CMakePlatform.LINUX_X86_64:
        x86_64_benchmarks.Linux_x86_64_Benchmarks.CASCADELAKE_COMPILE_CONFIG
}


@dataclass(frozen=True)
class ModelTestConfig(object):
  """Defines an e2e model test to run by iree-run-module."""
  # Test name shown in the test rule.
  name: str
  model: common_definitions.Model
  execution_config: iree_definitions.ModuleExecutionConfig

  # Either a string literal or a file path.
  expected_output: str
  input_data: common_definitions.ModelInputData = common_definitions.ZERO_MODEL_INPUT_DATA

  # Platforms to ignore this test.
  unsupported_platforms: List[CMakePlatform] = dataclasses.field(
      default_factory=list)
  # Platforms to expect this test failed.
  expected_fail_platforms: List[CMakePlatform] = dataclasses.field(
      default_factory=list)
  # Extra flags for `iree-run-module`.
  extra_test_flags: List[str] = dataclasses.field(default_factory=list)


TEST_CONFIGS = [
    # mobilenet_v1_fp32_correctness_test
    ModelTestConfig(
        name="mobilenet_v1_fp32_correctness_test",
        model=tflite_models.MOBILENET_V1,
        execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
        expected_output="mobilenet_v1_fp32_expected_output.txt",
        unsupported_platforms=[
            CMakePlatform.LINUX_RISCV32, CMakePlatform.ANDROID_ARMV8_A
        ]),
    # efficientnet_int8_correctness_test
    ModelTestConfig(
        name="efficientnet_int8_correctness_test",
        model=tflite_models.EFFICIENTNET_INT8,
        execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
        expected_output="efficientnet_int8_expected_output.txt",
        unsupported_platforms=[CMakePlatform.ANDROID_ARMV8_A]),
    # deeplab_v3_fp32_correctness_test
    # ModelTestConfig(
    #     name="deeplab_v3_fp32_correctness_test",
    #     model=tflite_models.DEEPLABV3_FP32,
    #     execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
    #     expected_output="deeplab_v3_fp32_input_0_expected_output.npy",
    #     extra_test_flags=["--expected_f32_threshold=0.001"],
    #     unsupported_platforms=[CMakePlatform.LINUX_RISCV32]),
    # person_detect_int8_correctness_test
    ModelTestConfig(
        name="person_detect_int8_correctness_test",
        model=tflite_models.PERSON_DETECT_INT8,
        execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
        expected_output="1x2xi8=[72 -72]",
        unsupported_platforms=[CMakePlatform.ANDROID_ARMV8_A])
]


def __get_iree_run_module_args(test_config: ModelTestConfig) -> List[str]:
  model = test_config.model
  args = [f"--entry_function={model.entry_function}"]
  if test_config.input_data.data_format != common_definitions.InputDataFormat.ZERO:
    raise ValueError("Currently only support all-zeros data.")
  args += [
      f"--function_input={input_type}=0" for input_type in model.input_types
  ]
  args += test_config.extra_test_flags
  return args


def generate_rules() -> List[str]:
  """Generates CMake rules for e2e model tests."""
  cmake_rules = []

  for platform in CMakePlatform:
    cmake_rule = cmake_builder.rules.build_set(
        variable_name=f"_MODULE_COMPILE_CONFIG_{platform.value.upper()}",
        value=f'"{PLATFORM_COMPILE_CONFIG_MAP[platform].id}"')
    cmake_rules.append(cmake_rule)

  for test_config in TEST_CONFIGS:
    model = test_config.model
    execution_config = test_config.execution_config
    cmake_rule = cmake_builder.rules.build_iree_benchmark_suite_module_test(
        target_name=test_config.name,
        model=f"{model.name}",
        driver=execution_config.driver.value,
        expected_output=test_config.expected_output,
        runner_args=__get_iree_run_module_args(test_config))
    cmake_rules.append(cmake_rule)

  return cmake_rules
