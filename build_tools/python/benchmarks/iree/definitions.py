## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE benchmarks."""

import itertools
from typing import List, Sequence, Tuple

from e2e_test_framework.device_specs import linux_x86_64_specs
from e2e_test_framework.models import model_groups
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework import unique_ids

MODULE_BENCHMARK_TOOL = "iree-benchmark-module"


def _generate_e2e_model_run_configs(
    model_compile_configs: Sequence[iree_definitions.ModelCompileConfig],
    vmfb_execution_configs: Sequence[iree_definitions.VMFBExecutionConfig],
    device_spec: common_definitions.DeviceSpec,
    input_data: common_definitions.ModelInputData = common_definitions.
    RANDOM_MODEL_INPUT_DATA,
) -> List[iree_definitions.E2EModelRunConfig]:
  """Generates the run configs from the product of compile configs and execution configs."""
  return [
      iree_definitions.E2EModelRunConfig(
          model_compile_config=model_compile_config,
          vmfb_execution_config=vmfb_execution_config,
          target_device_spec=device_spec,
          input_data=input_data)
      for model_compile_config, vmfb_execution_config in itertools.product(
          model_compile_configs, vmfb_execution_configs)
  ]


class Linux_x86_64_Benchmarks(object):
  """Benchmarks on x86_64 linux devices."""

  CASCADELAKE_CPU_TARGET = iree_definitions.CompileTarget(
      target_architecture=common_definitions.DeviceArchitecture.
      X86_64_CASCADELAKE,
      target_platform=common_definitions.DevicePlatform.LINUX_GNU,
      target_backend=iree_definitions.TargetBackend.LLVM_CPU)

  CASCADELAKE_COMPILE_CONFIG = iree_definitions.CompileConfig(
      id=unique_ids.IREE_COMPILE_CONFIG_LINUX_CASCADELAKE,
      tags=["default-flags"],
      compile_targets=[CASCADELAKE_CPU_TARGET])

  @classmethod
  def generate(
      cls
  ) -> Tuple[List[iree_definitions.ModelCompileConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    """Generates IREE compile and run configs."""

    default_execution_configs = cls._generate_default_execution_configs()

    model_compile_configs = [
        iree_definitions.ModelCompileConfig(
            compile_config=cls.CASCADELAKE_COMPILE_CONFIG, model=model)
        for model in model_groups.SMALL + model_groups.LARGE
    ]
    e2e_model_run_configs = _generate_e2e_model_run_configs(
        model_compile_configs=model_compile_configs,
        vmfb_execution_configs=default_execution_configs,
        device_spec=linux_x86_64_specs.GCP_C2_STANDARD_16)

    return (model_compile_configs, e2e_model_run_configs)

  @staticmethod
  def _generate_default_execution_configs(
  ) -> List[iree_definitions.VMFBExecutionConfig]:
    vmfb_execution_configs = [
        iree_definitions.VMFBExecutionConfig(
            id=unique_ids.IREE_VMFB_EXECUTION_CONFIG_LOCAL_SYNC,
            tags=["full-inference", "default-flags"],
            loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
            driver=iree_definitions.RuntimeDriver.LOCAL_SYNC,
            tool=MODULE_BENCHMARK_TOOL)
    ]
    for thread_num in [1, 4, 8]:
      vmfb_execution_configs.append(
          iree_definitions.VMFBExecutionConfig(
              id=
              f"{unique_ids.IREE_VMFB_EXECUTION_CONFIG_LOCAL_TASK_BASE}_{thread_num}",
              tags=[f"{thread_num}-thread", "full-inference", "default-flags"],
              loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
              driver=iree_definitions.RuntimeDriver.LOCAL_TASK,
              tool=MODULE_BENCHMARK_TOOL,
              extra_flags=[f"--task_topology_group_count={thread_num}"]))
    return vmfb_execution_configs


def generate() -> Tuple[List[iree_definitions.ModelCompileConfig],
                        List[iree_definitions.E2EModelRunConfig]]:
  """Generates all compile and run configs for IREE benchmarks."""
  return Linux_x86_64_Benchmarks.generate()
