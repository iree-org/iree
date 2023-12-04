## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE x86_64 benchmarks."""

from typing import List, Sequence

from benchmark_suites.iree import benchmark_presets, module_execution_configs, utils
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.device_specs import device_collections
from e2e_test_framework.models import model_groups
from e2e_test_framework import unique_ids


class Linux_x86_64_Benchmarks(object):
    """Benchmarks on x86_64 linux devices."""

    CASCADELAKE_CPU_TARGET = iree_definitions.CompileTarget(
        target_architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
        target_backend=iree_definitions.TargetBackend.LLVM_CPU,
        target_abi=iree_definitions.TargetABI.LINUX_GNU,
    )

    CASCADELAKE_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_LINUX_CASCADELAKE,
        tags=["default-flags"],
        compile_targets=[CASCADELAKE_CPU_TARGET],
    )
    CASCADELAKE_DATA_TILING_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_LINUX_CASCADELAKE_DATA_TILING,
        tags=["experimental-flags", "data-tiling", "ukernel"],
        compile_targets=[CASCADELAKE_CPU_TARGET],
        extra_flags=[
            "--iree-opt-data-tiling",
            "--iree-llvmcpu-enable-ukernels=all",
        ],
    )

    def _generate(
        self,
        benchmark_configs: List[common_definitions.CpuBenchmarkConfig],
        compile_config: iree_definitions.CompileConfig,
        device_specs: List[common_definitions.DeviceSpec],
        presets: Sequence[str],
    ) -> List[iree_definitions.E2EModelRunConfig]:
        run_configs_all = []
        # We avoid the full combinatorial explosion of testing all models with all
        # thread counts and instead test each model with a number of threads
        # appropriate for its size and configurations we're interested in.
        for config in benchmark_configs:
            gen_config = iree_definitions.ModuleGenerationConfig.build(
                compile_config=compile_config,
                imported_model=iree_definitions.ImportedModel.from_model(config.model),
            )

            execution_configs = []
            for thread in config.threads:
                if thread == 0:
                    execution_configs.append(
                        module_execution_configs.ELF_LOCAL_SYNC_CONFIG
                    )
                else:
                    execution_configs.append(
                        module_execution_configs.get_elf_local_task_config(thread)
                    )

            run_configs = utils.generate_e2e_model_run_configs(
                module_generation_configs=[gen_config],
                module_execution_configs=execution_configs,
                device_specs=device_specs,
                presets=presets,
            )

            run_configs_all.extend(run_configs)

        return run_configs_all

    def generate(
        self,
    ) -> List[iree_definitions.E2EModelRunConfig]:
        """Generates IREE compile and run configs."""

        cascadelake_devices = (
            device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
                architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
                host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
            )
        )

        # The X86_64 tag is required to put them into the X86_64 benchmark preset.
        default_run_configs = self._generate(
            model_groups.X86_64_BENCHMARK_CONFIG,
            self.CASCADELAKE_COMPILE_CONFIG,
            cascadelake_devices,
            presets=[benchmark_presets.X86_64],
        )
        experimental_run_configs = self._generate(
            model_groups.X86_64_BENCHMARK_CONFIG_EXPERIMENTAL,
            self.CASCADELAKE_DATA_TILING_COMPILE_CONFIG,
            cascadelake_devices,
            presets=[benchmark_presets.X86_64],
        )

        large_run_configs = self._generate(
            model_groups.X86_64_BENCHMARK_CONFIG_LONG,
            self.CASCADELAKE_COMPILE_CONFIG,
            cascadelake_devices,
            presets=[benchmark_presets.X86_64_LARGE],
        )

        return default_run_configs + experimental_run_configs + large_run_configs


def generate() -> List[iree_definitions.E2EModelRunConfig]:
    """Generates all compile and run configs for IREE benchmarks."""
    return Linux_x86_64_Benchmarks().generate()
