## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE VMVX benchmarks."""

from typing import List

from benchmark_suites.iree import benchmark_presets, module_execution_configs, utils
from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.device_specs import device_collections
from e2e_test_framework.models import tflite_models


class VMVX_Benchmarks(object):
    """Benchmarks with VMVX backend on different platforms."""

    VMVX_CPU_TARGET = iree_definitions.CompileTarget(
        target_backend=iree_definitions.TargetBackend.VMVX,
        target_architecture=common_definitions.DeviceArchitecture.VMVX_GENERIC,
        target_abi=iree_definitions.TargetABI.VMVX,
    )
    EXPERIMENTAL_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_VMVX_GENERIC_EXPERIMENTAL,
        tags=["experimental-flags"],
        compile_targets=[VMVX_CPU_TARGET],
    )

    def generate(
        self,
    ) -> List[iree_definitions.E2EModelRunConfig]:
        """Generates IREE compile and run configs."""

        gen_configs = [
            iree_definitions.ModuleGenerationConfig.build(
                compile_config=self.EXPERIMENTAL_COMPILE_CONFIG,
                imported_model=iree_definitions.ImportedModel.from_model(model),
            )
            for model in [tflite_models.MOBILENET_V2, tflite_models.MOBILENET_V3SMALL]
        ]
        local_task_exec_configs = [
            module_execution_configs.get_vmvx_local_task_config(thread_num=8)
        ]
        cascadelake_devices = (
            device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
                architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
                host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
            )
        )
        run_configs = utils.generate_e2e_model_run_configs(
            module_generation_configs=gen_configs,
            module_execution_configs=local_task_exec_configs,
            device_specs=cascadelake_devices,
            presets=[benchmark_presets.X86_64],
        )

        return run_configs
