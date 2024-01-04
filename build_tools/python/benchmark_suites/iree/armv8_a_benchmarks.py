## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE ARMv8-A benchmarks."""

from typing import List, Sequence

from benchmark_suites.iree import benchmark_presets, module_execution_configs, utils
from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.device_specs import pixel_8_pro_specs
from e2e_test_framework.models import tflite_models, tf_models


class Android_ARMv8_A_Benchmarks(object):
    """Benchmarks on ARMv8-A Android devices."""

    MODELS = [
        tflite_models.DEEPLABV3_FP32,
        tflite_models.MOBILEBERT_FP32,
        tf_models.GPT2_117M_1x4_FP32_TF,
        tf_models.GPT2_117M_1x1_FP32_TF,
        tflite_models.MOBILEBERT_INT8,
        tflite_models.VIT_INT8_TFL,
    ]

    ARMV8_A_CPU_TARGET = iree_definitions.CompileTarget(
        target_architecture=common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
        target_backend=iree_definitions.TargetBackend.LLVM_CPU,
        target_abi=iree_definitions.TargetABI.LINUX_ANDROID34,
    )

    PIXEL_8_CPU_FLAGS = "--iree-llvmcpu-target-cpu-features=" + ",".join(
        [
            "+v9a",
            "+fullfp16",
            "fp-armv8",
            "+neon",
            "+aes",
            "+sha2",
            "+crc",
            "+lse",
            "+rdm",
            "+complxnum",
            "+rcpc",
            "+sha3",
            "+sm4",
            "+dotprod",
            "+fp16fml",
            "+dit",
            "+flagm",
            "+ssbs",
            "+sb",
            "+sve2-aes",
            "+sve2-bitperm",
            "+sve2-sha3",
            "+sve2-sm4",
            "+altnzcv",
            "+fptoint",
            "+bf16",
            "+i8mm",
            "+bti",
            "+mte",
            "+pauth",
            "+perfmon",
            "+predres",
            "+spe",
            "+ras",
        ]
    )

    NO_DT_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_NO_DT,
        tags=["experimental-flags", "no-dt"],
        compile_targets=[ARMV8_A_CPU_TARGET],
        extra_flags=[
            "--iree-opt-data-tiling=false",
            PIXEL_8_CPU_FLAGS,
        ],
    )
    DT_ONLY_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_DT_ONLY,
        tags=["experimental-flags", "dt-only"],
        compile_targets=[ARMV8_A_CPU_TARGET],
        extra_flags=[
            "--iree-opt-data-tiling=true",
            "--iree-llvmcpu-enable-ukernels=none",
            PIXEL_8_CPU_FLAGS,
        ],
    )
    DT_UK_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARMV8_2_A_GENERIC_DT_UK,
        tags=["default-flags", "dt-uk"],
        compile_targets=[ARMV8_A_CPU_TARGET],
        extra_flags=[
            "--iree-opt-data-tiling=true",
            "--iree-llvmcpu-enable-ukernels=all",
            PIXEL_8_CPU_FLAGS,
        ],
    )

    def _build_run_configs(
        self,
        gen_configs: Sequence[iree_definitions.ModuleGenerationConfig],
        exec_configs: Sequence[iree_definitions.ModuleExecutionConfig],
        device_specs: Sequence[common_definitions.DeviceSpec],
        presets: Sequence[str],
    ) -> List[iree_definitions.E2EModelRunConfig]:
        return utils.generate_e2e_model_run_configs(
            module_generation_configs=gen_configs,
            module_execution_configs=exec_configs,
            device_specs=device_specs,
            presets=presets,
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
            for thread_num in [1, 5]
        ]

        no_dt_gen_confings = [
            iree_definitions.ModuleGenerationConfig.build(
                compile_config=self.NO_DT_COMPILE_CONFIG,
                imported_model=iree_definitions.ImportedModel.from_model(model),
            )
            for model in self.MODELS
        ]
        dt_only_gen_confings = [
            iree_definitions.ModuleGenerationConfig.build(
                compile_config=self.DT_ONLY_COMPILE_CONFIG,
                imported_model=iree_definitions.ImportedModel.from_model(model),
            )
            for model in self.MODELS
        ]
        dt_uk_gen_confings = [
            iree_definitions.ModuleGenerationConfig.build(
                compile_config=self.DT_UK_COMPILE_CONFIG,
                imported_model=iree_definitions.ImportedModel.from_model(model),
            )
            for model in self.MODELS
        ]

        big_cores_devices = [pixel_8_pro_specs.BIG_CORES]
        run_configs = self._build_run_configs(
            no_dt_gen_confings + dt_uk_gen_confings,
            local_sync_execution_configs + local_task_execution_configs,
            big_cores_devices,
            [benchmark_presets.ANDROID_CPU],
        ) + self._build_run_configs(
            dt_only_gen_confings,
            local_sync_execution_configs + local_task_execution_configs,
            big_cores_devices,
            [benchmark_presets.ANDROID_CPU_DT_ONLY],
        )

        return run_configs
