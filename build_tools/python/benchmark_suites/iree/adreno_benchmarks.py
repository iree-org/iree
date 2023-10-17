## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE Adreno GPU benchmarks."""

from typing import List

from benchmark_suites.iree import benchmark_presets, module_execution_configs, utils
from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.models import tflite_models, tf_models
from e2e_test_framework.device_specs import device_collections


class Android_Adreno_Benchmarks(object):
    """Benchmarks on Android devices with Adreno GPU."""

    ADRENO_GPU_COMPILE_TARGET = iree_definitions.CompileTarget(
        target_backend=iree_definitions.TargetBackend.VULKAN_SPIRV,
        target_architecture=common_definitions.DeviceArchitecture.QUALCOMM_ADRENO,
        target_abi=iree_definitions.TargetABI.VULKAN_ANDROID31,
    )
    DEFAULT_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_QUALCOMM_ADRENO_DEFAULTS,
        tags=["default-flags"],
        compile_targets=[ADRENO_GPU_COMPILE_TARGET],
    )
    FUSE_PADDING_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_QUALCOMM_ADRENO_FUSE_PADDING,
        tags=["experimental-flags", "fuse-padding"],
        compile_targets=[ADRENO_GPU_COMPILE_TARGET],
        extra_flags=["--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"],
    )
    FUSE_PADDING_REPEATED_KERNEL_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_QUALCOMM_ADRENO_FUSE_PADDING_REPEATED_KERNEL,
        tags=["experimental-flags", "fuse-padding", "repeated-kernel"],
        compile_targets=[ADRENO_GPU_COMPILE_TARGET],
        extra_flags=FUSE_PADDING_COMPILE_CONFIG.extra_flags
        + ["--iree-hal-benchmark-dispatch-repeat-count=16"],
    )

    def generate(
        self,
    ) -> List[iree_definitions.E2EModelRunConfig]:
        default_models = [
            tflite_models.MOBILEBERT_FP32,
        ]
        default_gen_configs = [
            iree_definitions.ModuleGenerationConfig.build(
                compile_config=self.DEFAULT_COMPILE_CONFIG,
                imported_model=iree_definitions.ImportedModel.from_model(model),
            )
            for model in default_models
        ]
        fuse_padding_gen_configs = [
            iree_definitions.ModuleGenerationConfig.build(
                compile_config=self.FUSE_PADDING_COMPILE_CONFIG,
                imported_model=iree_definitions.ImportedModel.from_model(model),
            )
            for model in default_models
        ]

        adreno_devices = (
            device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
                architecture=common_definitions.DeviceArchitecture.QUALCOMM_ADRENO,
                host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
            )
        )
        run_configs = utils.generate_e2e_model_run_configs(
            module_generation_configs=default_gen_configs,
            module_execution_configs=[module_execution_configs.VULKAN_CONFIG],
            device_specs=adreno_devices,
            presets=[benchmark_presets.ANDROID_GPU],
        )
        run_configs += utils.generate_e2e_model_run_configs(
            module_generation_configs=fuse_padding_gen_configs,
            module_execution_configs=[module_execution_configs.VULKAN_CONFIG],
            device_specs=adreno_devices,
            presets=[benchmark_presets.ANDROID_GPU],
        )

        return run_configs
