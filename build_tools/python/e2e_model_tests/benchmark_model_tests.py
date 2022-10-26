## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines e2e tests for modules in benchmark suites."""

from dataclasses import dataclass
from typing import List, Sequence, Set
import dataclasses
import enum

from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.definitions import iree_definitions
from e2e_test_framework.models import tflite_models
from benchmark_suites.iree import (riscv_benchmarks, x86_64_benchmarks,
                                   armv8_a_benchmarks, module_execution_configs)

# iree_definitions.ImportedModel.from_model(tflite_models.MOBILENET_V1)


class CMAKE_SYSTEM_PLATFORM(enum.Enum):
  """Enum of CMake detected system platform string."""
  ANDROID_ARMV8_A = "android-arm64-v8a"
  LINUX_RISCV32 = "riscv32-Linux"
  LINUX_RISCV64 = "riscv64-Linux"
  LINUX_X86_64 = "x86-64-Linux"


PLATFORM_COMPILE_CONFIG_MAP = {
    CMAKE_SYSTEM_PLATFORM.ANDROID_ARMV8_A:
        armv8_a_benchmarks.Android_ARMv8_A_Benchmarks.DEFAULT_COMPILE_CONFIG,
    CMAKE_SYSTEM_PLATFORM.LINUX_RISCV32:
        riscv_benchmarks.Linux_RV32_Benchmarks.DEFAULT_COMPILE_CONFIG,
    CMAKE_SYSTEM_PLATFORM.LINUX_RISCV64:
        riscv_benchmarks.Linux_RV64_Benchmarks.DEFAULT_COMPILE_CONFIG,
    CMAKE_SYSTEM_PLATFORM.LINUX_X86_64:
        x86_64_benchmarks.Linux_x86_64_Benchmarks.CASCADELAKE_COMPILE_CONFIG
}


@dataclass(frozen=True)
class ModelTestConfig(object):
  """Defines an e2e model test to run by iree-run-module."""
  # Test name shown in the test rule.
  name: str
  module_generation_config: iree_definitions.ModuleGenerationConfig
  module_execution_config: iree_definitions.ModuleExecutionConfig
  input_data: common_definitions.ModelInputData
  # Can be either a string literal or "@{file path}".
  expected_output: str
  # Extra flags for `iree-run-module`.
  extra_test_flags: List[str] = dataclasses.field(default_factory=list)


def _generate_model_test_configs(
    name: str,
    model: common_definitions.Model,
    module_execution_config: iree_definitions.ModuleExecutionConfig,
    expected_output: str,
    extra_test_flags: Sequence[str] = [],
    excluded_platforms: Set[CMAKE_SYSTEM_PLATFORM] = set()
) -> List[ModelTestConfig]:
  """Finds the matched generation configs and composes the E2EModuleTestConfig.
  """

  test_configs = []
  for platform, compile_config in PLATFORM_COMPILE_CONFIG_MAP.items():
    if platform in excluded_platforms:
      continue

    module_generation_config = iree_definitions.ModuleGenerationConfig(
        imported_model=iree_definitions.ImportedModel.from_model(model),
        compile_config=compile_config)
    test_configs.append(
        ModelTestConfig(name=name,
                        module_generation_config=module_generation_config,
                        module_execution_config=module_execution_config,
                        input_data=common_definitions.ZERO_MODEL_INPUT_DATA,
                        expected_output=expected_output,
                        extra_test_flags=list(extra_test_flags)))

  return test_configs


def generate_model_tests() -> List[ModelTestConfig]:
  """Generates e2e model correctness tests based on default compile configs."""

  test_configs = []

  # mobilenet_v1_fp32_correctness_test
  test_configs += _generate_model_test_configs(
      name="mobilenet_v1_fp32_correctness_test",
      model=tflite_models.MOBILENET_V1,
      module_execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
      expected_output="@mobilenet_v1_fp32_expected_output.txt",
      excluded_platforms={CMAKE_SYSTEM_PLATFORM.LINUX_RISCV32})

  # efficientnet_int8_correctness_test
  test_configs += _generate_model_test_configs(
      name="efficientnet_int8_correctness_test",
      model=tflite_models.EFFICIENTNET_INT8,
      module_execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
      expected_output="@efficientnet_int8_expected_output.txt")

  # deeplab_v3_fp32_correctness_test
  test_configs += _generate_model_test_configs(
      name="deeplab_v3_fp32_correctness_test",
      model=tflite_models.DEEPLABV3_FP32,
      module_execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
      expected_output="@deeplab_v3_fp32_input_0_expected_output.npy",
      extra_test_flags=["--expected_f32_threshold=0.001"],
      excluded_platforms={CMAKE_SYSTEM_PLATFORM.LINUX_RISCV32})

  # person_detect_int8_correctness_test
  test_configs += _generate_model_test_configs(
      name="person_detect_int8_correctness_test",
      model=tflite_models.PERSON_DETECT_INT8,
      module_execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
      expected_output="1x2xi8=[72 -72]")

  return test_configs
