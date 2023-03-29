## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE CUDA benchmarks."""

from typing import List, Tuple
from benchmark_suites.iree import module_execution_configs
from e2e_test_framework.models import torch_models
from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.device_specs import device_collections
from e2e_test_framework.models import model_groups
import benchmark_suites.iree.utils


class Linux_CUDA_Benchmarks(object):
  """Benchmarks on CUDA Linux devices."""

  SM_80_GPU_TARGET = iree_definitions.CompileTarget(
      target_architecture=common_definitions.DeviceArchitecture.CUDA_SM80,
      target_backend=iree_definitions.TargetBackend.CUDA,
      target_abi=iree_definitions.TargetABI.LINUX_GNU)
  SM_80_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
      id=unique_ids.IREE_COMPILE_CONFIG_LINUX_CUDA_SM80_DEFAULTS,
      tags=["default-flags"],
      compile_targets=[SM_80_GPU_TARGET])

  def generate(
      self
  ) -> Tuple[List[iree_definitions.ModuleGenerationConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    """Generates IREE compile and run configs."""
    models = model_groups.LARGE
    gen_configs = [
        iree_definitions.ModuleGenerationConfig.build(
            compile_config=self.SM_80_COMPILE_CONFIG,
            imported_model=iree_definitions.ImportedModel.from_model(model))
        for model in models
    ]
    sm80_devices = device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.CUDA_SM80,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64)
    run_module_configs = benchmark_suites.iree.utils.generate_e2e_model_run_configs(
        module_generation_configs=gen_configs,
        module_execution_configs=[module_execution_configs.CUDA_CONFIG],
        device_specs=sm80_devices)

    return (gen_configs, run_module_configs)
