## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE ARMv8-A benchmarks."""

from typing import List

from benchmark_suites.iree import benchmark_presets, module_execution_configs, utils
from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.device_specs import device_collections
from e2e_test_framework.models import tflite_models, tf_models


class Android_ARMv8_A_Benchmarks(object):
    """Benchmarks on ARMv8-A Android devices."""

    NONQUANT_MODELS = [
        tflite_models.DEEPLABV3_FP32,
        tflite_models.MOBILEBERT_FP32,
        tf_models.GPT2_117M_1x4_FP32_TF,
        tf_models.GPT2_117M_1x1_FP32_TF,
    ]
    QUANT_MODELS = [
        tflite_models.MOBILEBERT_INT8,
        tflite_models.VIT_INT8_TFL,
    ]

    ARMV8_A_CPU_TARGET = iree_definitions.CompileTarget(
        target_architecture=common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
        target_backend=iree_definitions.TargetBackend.LLVM_CPU,
        target_abi=iree_definitions.TargetABI.LINUX_ANDROID29,
    )

    DEFAULT_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_DEFAULTS,
        tags=["default-flags"],
        compile_targets=[ARMV8_A_CPU_TARGET],
    )
    DATA_TILING_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_MMT4D,
        tags=["experimental-flags", "data-tiling", "ukernel"],
        compile_targets=[ARMV8_A_CPU_TARGET],
        extra_flags=[
            "--iree-opt-data-tiling",
            "--iree-llvmcpu-enable-ukernels=all",
        ],
    )
    DATA_TILING_AND_DOTPROD_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_MMT4D_DOTPROD,
        tags=["experimental-flags", "data-tiling", "ukernel", "dotprod"],
        compile_targets=[ARMV8_A_CPU_TARGET],
        extra_flags=[
            "--iree-opt-data-tiling",
            "--iree-llvmcpu-enable-ukernels=all",
            "--iree-llvmcpu-target-cpu-features=+dotprod",
        ],
    )

    def generate(
        self,
    ) -> List[iree_definitions.E2EModelRunConfig]:
        """Generates IREE compile and run configs."""

        local_sync_execution_configs = [module_execution_configs.ELF_LOCAL_SYNC_CONFIG]
        local_task_execution_configs = [
            module_execution_configs.get_elf_system_scheduling_local_task_config(
                thread_num
            )
            for thread_num in [1, 2]
        ]

        default_gen_confings = [
            iree_definitions.ModuleGenerationConfig.build(
                compile_config=self.DEFAULT_COMPILE_CONFIG,
                imported_model=iree_definitions.ImportedModel.from_model(model),
            )
            for model in self.NONQUANT_MODELS + self.QUANT_MODELS
        ]
        experimental_gen_confings = [
            iree_definitions.ModuleGenerationConfig.build(
                compile_config=self.DATA_TILING_COMPILE_CONFIG,
                imported_model=iree_definitions.ImportedModel.from_model(model),
            )
            for model in self.NONQUANT_MODELS
        ] + [
            iree_definitions.ModuleGenerationConfig.build(
                compile_config=self.DATA_TILING_AND_DOTPROD_COMPILE_CONFIG,
                imported_model=iree_definitions.ImportedModel.from_model(model),
            )
            for model in self.QUANT_MODELS
        ]

        big_cores_devices = (
            device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
                architecture=common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
                host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
                tags=["big-cores"],
            )
        )
        run_configs = utils.generate_e2e_model_run_configs(
            module_generation_configs=default_gen_confings,
            module_execution_configs=local_sync_execution_configs
            + local_task_execution_configs,
            device_specs=big_cores_devices,
            presets=[benchmark_presets.ANDROID_CPU],
        )
        run_configs += utils.generate_e2e_model_run_configs(
            module_generation_configs=experimental_gen_confings,
            module_execution_configs=local_task_execution_configs,
            device_specs=big_cores_devices,
            presets=[benchmark_presets.ANDROID_CPU],
        )

        return run_configs
