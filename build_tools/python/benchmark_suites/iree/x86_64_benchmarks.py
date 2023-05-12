## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE x86_64 benchmarks."""

from typing import List, Sequence, Tuple
from benchmark_suites.iree import benchmark_tags
from e2e_test_framework.device_specs import device_collections
from e2e_test_framework.models import model_groups, tf_models, torch_models
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

  CASCADELAKE_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
      id=unique_ids.IREE_COMPILE_CONFIG_LINUX_CASCADELAKE,
      tags=["default-flags"],
      compile_targets=[CASCADELAKE_CPU_TARGET])
  CASCADELAKE_FUSE_PADDING_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
      id=unique_ids.IREE_COMPILE_CONFIG_LINUX_CASCADELAKE_FUSE_PADDING,
      tags=["experimental-flags", "fuse-padding"],
      compile_targets=[CASCADELAKE_CPU_TARGET],
      extra_flags=[
          "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops",
          "--iree-llvmcpu-enable-pad-consumer-fusion"
      ])

  def _generate(
      self,
      benchmark_configs: List[common_definitions.CpuBenchmarkConfig],
      compile_config: iree_definitions.CompileConfig,
      device_specs: List[common_definitions.DeviceSpec],
      tags: Sequence[str] = [],
  ) -> Tuple[List[iree_definitions.ModuleGenerationConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    gen_configs_all = []
    run_configs_all = []

    # We avoid the full combinatorial explosion of testing all models with all
    # thread counts and instead test each model with a number of threads
    # appropriate for its size and configurations we're interested in.
    for config in benchmark_configs:
      gen_config = iree_definitions.ModuleGenerationConfig.build(
          compile_config=compile_config,
          imported_model=iree_definitions.ImportedModel.from_model(
              config.model),
          tags=tags)

      execution_configs = []
      for thread in config.threads:
        if thread == 0:
          execution_configs.append(
              module_execution_configs.ELF_LOCAL_SYNC_CONFIG)
        else:
          execution_configs.append(
              module_execution_configs.get_elf_local_task_config(thread))

      run_configs = benchmark_suites.iree.utils.generate_e2e_model_run_configs(
          module_generation_configs=[gen_config],
          module_execution_configs=execution_configs,
          device_specs=device_specs,
          tags=tags)

      gen_configs_all.append(gen_config)
      run_configs_all.extend(run_configs)

    return (gen_configs_all, run_configs_all)

  def generate(
      self
  ) -> Tuple[List[iree_definitions.ModuleGenerationConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    """Generates IREE compile and run configs."""

    cascadelake_devices = device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64)

    # The X86_64 tag is required to put them into the X86_64 benchmark preset.
    default_gen_configs, default_run_configs = self._generate(
        model_groups.X86_64_BENCHMARK_CONFIG,
        self.CASCADELAKE_COMPILE_CONFIG,
        cascadelake_devices,
        tags=[benchmark_tags.X86_64])
    experimental_gen_configs, experimental_run_configs = self._generate(
        model_groups.X86_64_BENCHMARK_CONFIG_EXPERIMENTAL,
        self.CASCADELAKE_FUSE_PADDING_COMPILE_CONFIG,
        cascadelake_devices,
        tags=[benchmark_tags.X86_64])

    long_running_gen_configs, long_running_run_configs = self._generate(
        model_groups.X86_64_BENCHMARK_CONFIG_LONG,
        self.CASCADELAKE_COMPILE_CONFIG,
        cascadelake_devices,
        tags=[benchmark_tags.X86_64, benchmark_tags.LONG_RUNNING])

    return (default_gen_configs + experimental_gen_configs +
            long_running_gen_configs, default_run_configs +
            experimental_run_configs + long_running_run_configs)


def generate() -> Tuple[List[iree_definitions.ModuleGenerationConfig],
                        List[iree_definitions.E2EModelRunConfig]]:
  """Generates all compile and run configs for IREE benchmarks."""
  return Linux_x86_64_Benchmarks().generate()
