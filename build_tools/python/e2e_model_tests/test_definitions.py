## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines e2e model tests on benchmark models."""

from typing import List
from dataclasses import dataclass
import dataclasses
import enum

from e2e_test_framework.definitions import iree_definitions
from e2e_test_framework.models import tflite_models
from benchmark_suites.iree import (
    arm64_benchmarks,
    riscv_benchmarks,
    x86_64_benchmarks,
    module_execution_configs,
)


class CMakePlatform(enum.Enum):
    """Enum of CMake system platform string."""

    ANDROID_ARM64 = "arm_64-Android"
    LINUX_RISCV32 = "riscv_32-Linux"
    LINUX_RISCV64 = "riscv_64-Linux"
    LINUX_X86_64 = "x86_64-Linux"


# Compile config used for each CMake system platform.
PLATFORM_COMPILE_CONFIG_MAP = {
    CMakePlatform.ANDROID_ARM64: arm64_benchmarks.Android_ARM64_Benchmarks.DT_UK_COMPILE_CONFIG,
    CMakePlatform.LINUX_RISCV32: riscv_benchmarks.Linux_RV32_Benchmarks.DEFAULT_COMPILE_CONFIG,
    CMakePlatform.LINUX_RISCV64: riscv_benchmarks.Linux_RV64_Benchmarks.DEFAULT_COMPILE_CONFIG,
    CMakePlatform.LINUX_X86_64: x86_64_benchmarks.Linux_x86_64_Benchmarks.CASCADELAKE_DT_UK_COMPILE_CONFIG,
}


@dataclass(frozen=True)
class ModelTestConfig(object):
    """Defines an e2e model test to run by iree-run-module."""

    # Test name shown in the test rule.
    name: str
    imported_model: iree_definitions.ImportedModel
    execution_config: iree_definitions.ModuleExecutionConfig

    # Either a string literal or a file path.
    expected_output: str

    # Platforms to ignore this test.
    unsupported_platforms: List[CMakePlatform] = dataclasses.field(default_factory=list)
    # Platforms to expect this test failed.
    xfail_platforms: List[CMakePlatform] = dataclasses.field(default_factory=list)
    # Extra flags for `iree-run-module`.
    extra_test_flags: List[str] = dataclasses.field(default_factory=list)


TEST_CONFIGS = [
    # mobilenet_v1_fp32_correctness_test
    ModelTestConfig(
        name="mobilenet_v1_fp32_correctness_test",
        imported_model=iree_definitions.ImportedModel.from_model(
            tflite_models.MOBILENET_V1
        ),
        execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
        expected_output="mobilenet_v1_fp32_expected_output.txt",
        unsupported_platforms=[
            CMakePlatform.LINUX_RISCV32,
            CMakePlatform.ANDROID_ARM64,
        ],
    ),
    # efficientnet_int8_correctness_test
    ModelTestConfig(
        name="efficientnet_int8_correctness_test",
        imported_model=iree_definitions.ImportedModel.from_model(
            tflite_models.EFFICIENTNET_INT8
        ),
        execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
        expected_output="efficientnet_int8_expected_output.txt",
        unsupported_platforms=[
            CMakePlatform.ANDROID_ARM64,
            CMakePlatform.LINUX_RISCV32,
            CMakePlatform.LINUX_RISCV64,
        ],
    ),
    # deeplab_v3_fp32_correctness_test
    ModelTestConfig(
        name="deeplab_v3_fp32_correctness_test",
        imported_model=iree_definitions.ImportedModel.from_model(
            tflite_models.DEEPLABV3_FP32
        ),
        execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
        expected_output="https://storage.googleapis.com/iree-model-artifacts/deeplab_v3_fp32_input_0_expected_output.npy",
        extra_test_flags=["--expected_f32_threshold=0.001"],
        unsupported_platforms=[
            CMakePlatform.LINUX_RISCV32,
            CMakePlatform.LINUX_RISCV64,
        ],
    ),
    # person_detect_int8_correctness_test
    ModelTestConfig(
        name="person_detect_int8_correctness_test",
        imported_model=iree_definitions.ImportedModel.from_model(
            tflite_models.PERSON_DETECT_INT8
        ),
        execution_config=module_execution_configs.ELF_LOCAL_SYNC_CONFIG,
        expected_output="1x2xi8=[72 -72]",
        unsupported_platforms=[CMakePlatform.ANDROID_ARM64],
    ),
]
