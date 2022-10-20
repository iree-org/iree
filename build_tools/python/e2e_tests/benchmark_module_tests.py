## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines e2e tests for modules in benchmark suites."""

from typing import List, Sequence, Set
from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.definitions import iree_definitions
from e2e_tests import e2e_tests_definitions
from e2e_test_framework import unique_ids
from benchmark_suites.iree import benchmark_collections, module_execution_configs


def generate_e2e_module_test_config(
    name: str,
    source_generation_configs: Sequence[
        iree_definitions.ModuleGenerationConfig],
    match_model_id: str,
    match_compile_tags: Set[str],
    excluded_architectures: Set[common_definitions.DeviceArchitecture],
    module_execution_config: iree_definitions.ModuleExecutionConfig,
    expected_output: str,
    input_data: common_definitions.ModelInputData = common_definitions.
    ZERO_MODEL_INPUT_DATA,
    extra_test_flags: List[str] = []):
  """Find the matched generation configs and compose the E2EModuleTestConfig."""

  test_configs = []
  for gen_config in source_generation_configs:
    if gen_config.model.source_model.id != match_model_id:
      continue
    compile_config = gen_config.compile_config
    if not match_compile_tags <= set(compile_config.tags):
      continue
    # Skip if all target architectures are excluded from the test.
    if excluded_architectures >= set(
        target.target_architecture
        for target in compile_config.compile_targets):
      continue

    test_configs.append(
        e2e_tests_definitions.E2EModuleTestConfig(
            name=name,
            module_generation_config=gen_config,
            module_execution_config=module_execution_config,
            input_data=input_data,
            expected_output=expected_output,
            extra_test_flags=extra_test_flags))

  return test_configs


def generate_tests() -> List[e2e_tests_definitions.E2EModuleTestConfig]:
  (module_gen_configs, _) = benchmark_collections.generate_benchmarks()
  test_configs = []
  test_configs += generate_e2e_module_test_config(
      name="mobilenet_v1_fp32_correctness_test",
      source_generation_configs=module_gen_configs,
      match_model_id=unique_ids.MODEL_MOBILENET_V1,
      match_compile_tags={"default-flags"},
      excluded_architectures={
          common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
          common_definitions.DeviceArchitecture.RV32_GENERIC
      },
      module_execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
      expected_output="@mobilenet_v1_fp32_expected_output.txt")
  test_configs += generate_e2e_module_test_config(
      name="efficientnet_int8_correctness_test",
      source_generation_configs=module_gen_configs,
      match_model_id=unique_ids.MODEL_EFFICIENTNET_INT8,
      match_compile_tags={"default-flags"},
      excluded_architectures={
          common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
      },
      module_execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
      expected_output="@efficientnet_int8_expected_output.txt")
  test_configs += generate_e2e_module_test_config(
      name="deeplab_v3_fp32_correctness_test",
      source_generation_configs=module_gen_configs,
      match_model_id=unique_ids.MODEL_PERSON_DETECT_INT8,
      match_compile_tags={"default-flags"},
      excluded_architectures={
          common_definitions.DeviceArchitecture.RV32_GENERIC,
      },
      module_execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
      expected_output="@deeplab_v3_fp32_input_0_expected_output.npy",
      extra_test_flags=["--expected_f32_threshold=0.001"])
  test_configs += generate_e2e_module_test_config(
      name="person_detect_int8_correctness_test",
      source_generation_configs=module_gen_configs,
      match_model_id=unique_ids.MODEL_PERSON_DETECT_INT8,
      match_compile_tags={"default-flags"},
      excluded_architectures={
          common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
      },
      module_execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
      expected_output="1x2xi8=[72 -72]")
  return test_configs
