## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE Vulkan NVIDIA benchmarks."""

from typing import List, Tuple, Sequence
from benchmark_suites.iree import module_execution_configs
from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.device_specs import device_collections
from e2e_test_framework.models import model_groups
import benchmark_suites.iree.utils


def _get_compile_flag():
  preprocess_passes = [
      "iree-flow-detach-elementwise-from-named-ops",
      "iree-preprocessing-convert-conv2d-to-img2col",
      "iree-flow-convert-1x1-filter-conv2d-to-matmul",
      "iree-preprocessing-pad-linalg-ops{pad-size=32}",
  ]
  preprocess_flag_template = \
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func({}))"
  return [
      "--iree-stream-resource-index-bits=64", "--iree-vm-target-index-bits=64",
      preprocess_flag_template.format(",".join(preprocess_passes))
  ]


class Linux_Vulkan_NVIDIA_Benchmarks(object):
  """Benchmarks on Linux Vulkan NVIDIA devices."""

  AMPERE_TARGET = iree_definitions.CompileTarget(
      target_architecture=common_definitions.DeviceArchitecture.NVIDIA_AMPERE,
      target_backend=iree_definitions.TargetBackend.VULKAN_SPIRV,
      target_abi=iree_definitions.TargetABI.VULKAN_LINUX)
  PASCAL_TARGET = iree_definitions.CompileTarget(
      target_architecture=common_definitions.DeviceArchitecture.NVIDIA_PASCAL,
      target_backend=iree_definitions.TargetBackend.VULKAN_SPIRV,
      target_abi=iree_definitions.TargetABI.VULKAN_LINUX)

  SIMT_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
      id=unique_ids.IREE_COMPILE_CONFIG_LINUX_VULKAN_SD_SIMT,
      tags=["experimental-flags", "simt"],
      compile_targets=[PASCAL_TARGET],
      extra_flags=_get_compile_flag())
  TENSORCORE_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
      id=unique_ids.IREE_COMPILE_CONFIG_LINUX_VULKAN_SD_TENSORCORE,
      tags=["experimental-flags", "tensorcore"],
      compile_targets=[AMPERE_TARGET],
      extra_flags=_get_compile_flag())

  def _generate_configs(
      self,
      models: Sequence[common_definitions.Model],
      compile_config: iree_definitions.CompileConfig,
      execution_config: iree_definitions.
      ModuleExecutionConfig = module_execution_configs.VULKAN_CONFIG,
      run_tags: Sequence[str] = [],
  ) -> Tuple[List[iree_definitions.ModuleGenerationConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    gen_configs = [
        iree_definitions.ModuleGenerationConfig.build(
            compile_config=compile_config,
            imported_model=iree_definitions.ImportedModel.from_model(model))
        for model in models
    ]
    # We use the same NVIDIA Ampere GPU for benchmarking code generated for
    # both Pascal and Ampere architectures. What we care is not exactly these
    # two architectures per se; they represent SIMT and tensorcore CodeGen
    # paths that we would want both to work. Ampere is able to run both SIMT
    # and tensorcore cases.
    ampere_devices = device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.NVIDIA_AMPERE,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64)
    run_module_configs = benchmark_suites.iree.utils.generate_e2e_model_run_configs(
        module_generation_configs=gen_configs,
        module_execution_configs=[execution_config],
        device_specs=ampere_devices,
        tags=run_tags)

    return (gen_configs, run_module_configs)

  def generate(
      self
  ) -> Tuple[List[iree_definitions.ModuleGenerationConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    """Generates IREE compile and run configs."""
    tensorcore_gen_configs, tensorcore_run_configs = self._generate_configs(
        model_groups.VULKAN_MODELS, self.TENSORCORE_COMPILE_CONFIG)
    simt_gen_configs, simt_run_configs = self._generate_configs(
        model_groups.VULKAN_MODELS, self.SIMT_COMPILE_CONFIG)
    return (tensorcore_gen_configs + simt_gen_configs,
            tensorcore_run_configs + simt_run_configs)
