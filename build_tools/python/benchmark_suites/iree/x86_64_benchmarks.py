## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE x86_64 benchmarks."""

from typing import List, Tuple
from e2e_test_framework.device_specs import device_collections
from e2e_test_framework.models import model_groups, tf_models
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework import unique_ids
from benchmark_suites.iree import module_execution_configs
import benchmark_suites.iree.utils


class Linux_x86_64_Benchmarks(object):
  """Benchmarks on x86_64 linux devices."""

  CASCADELAKE_CPU_TARGET = iree_definitions.CompileTarget(
      target_architecture=common_definitions.DeviceArchitecture.
      X86_64_CASCADELAKE,
      target_backend=iree_definitions.TargetBackend.LLVM_CPU,
      target_abi=iree_definitions.TargetABI.LINUX_GNU)

  CASCADELAKE_COMPILE_CONFIG = iree_definitions.CompileConfig(
      id=unique_ids.IREE_COMPILE_CONFIG_LINUX_CASCADELAKE,
      tags=["default-flags"],
      compile_targets=[CASCADELAKE_CPU_TARGET])
  CASCADELAKE_FUSE_PADDING_COMPILE_CONFIG = iree_definitions.CompileConfig(
      id=unique_ids.IREE_COMPILE_CONFIG_LINUX_CASCADELAKE_FUSE_PADDING,
      tags=["experimental-flags", "fuse-padding"],
      compile_targets=[CASCADELAKE_CPU_TARGET],
      extra_flags=[
          "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops",
          "--iree-llvmcpu-enable-pad-consumer-fusion"
      ])

  def generate(
      self
  ) -> Tuple[List[iree_definitions.ModuleGenerationConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    """Generates IREE compile and run configs."""

    gen_configs = [
        iree_definitions.ModuleGenerationConfig.with_flag_generation(
            compile_config=self.CASCADELAKE_COMPILE_CONFIG,
            imported_model=iree_definitions.ImportedModel.from_model(model))
        for model in model_groups.SMALL + model_groups.LARGE
    ]
    # TODO(#11174): Excludes ResNet50
    excluded_models_for_experiments = [tf_models.RESNET50_TF_FP32]
    gen_configs += [
        iree_definitions.ModuleGenerationConfig.with_flag_generation(
            compile_config=self.CASCADELAKE_FUSE_PADDING_COMPILE_CONFIG,
            imported_model=iree_definitions.ImportedModel.from_model(model))
        for model in model_groups.SMALL + model_groups.LARGE
        if model not in excluded_models_for_experiments
    ]
    default_execution_configs = [
        module_execution_configs.ELF_LOCAL_SYNC_CONFIG
    ] + [
        module_execution_configs.get_elf_local_task_config(thread_num)
        for thread_num in [1, 4, 8]
    ]
    cascadelake_devices = device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64)
    run_configs = benchmark_suites.iree.utils.generate_e2e_model_run_configs(
        module_generation_configs=gen_configs,
        module_execution_configs=default_execution_configs,
        device_specs=cascadelake_devices)

    return (gen_configs, run_configs)


def generate() -> Tuple[List[iree_definitions.ModuleGenerationConfig],
                        List[iree_definitions.E2EModelRunConfig]]:
  """Generates all compile and run configs for IREE benchmarks."""
  return Linux_x86_64_Benchmarks().generate()
