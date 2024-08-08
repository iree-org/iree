# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers_tools import *
import os
import setuptools
from conftest import VmfbManager
from pathlib import Path

iree_test_path_extension = os.getenv("IREE_TEST_PATH_EXTENSION", default=Path.cwd())
rocm_chip = os.getenv("ROCM_CHIP", default="gfx90a")

###############################################################################
# Fixtures
###############################################################################

sdxl_unet_inference_input_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/inference_input.0.bin",
    group="sdxl_unet",
)

sdxl_unet_inference_input_1 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/inference_input.1.bin",
    group="sdxl_unet",
)

sdxl_unet_inference_input_2 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/inference_input.2.bin",
    group="sdxl_unet",
)

sdxl_unet_inference_input_3 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/inference_input.3.bin",
    group="sdxl_unet",
)

sdxl_unet_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/inference_output.0.bin",
    group="sdxl_unet",
)

sdxl_unet_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/real_weights.irpa",
    group="sdxl_unet",
)

sdxl_unet_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/model.mlirbc",
    group="sdxl_unet",
)

sdxl_unet_pipeline_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/sdxl_unet_pipeline_bench_f16.mlir",
    group="sdxl_unet",
)

CPU_COMPILE_FLAGS = [
    "--iree-hal-target-backends=llvm-cpu",
    "--iree-llvmcpu-target-cpu-features=host",
    "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false",
    "--iree-llvmcpu-distribution-size=32",
    "--iree-opt-const-eval=false",
    "--iree-llvmcpu-enable-ukernels=all",
    "--iree-global-opt-enable-quantized-matmul-reassociation",
]


@pytest.fixture
def SDXL_UNET_COMMON_RUN_FLAGS(
    sdxl_unet_inference_input_0,
    sdxl_unet_inference_input_1,
    sdxl_unet_inference_input_2,
    sdxl_unet_inference_input_3,
    sdxl_unet_inference_output_0,
):
    return [
        f"--input=1x4x128x128xf16=@{sdxl_unet_inference_input_0.path}",
        f"--input=2x64x2048xf16=@{sdxl_unet_inference_input_1.path}",
        f"--input=2x1280xf16=@{sdxl_unet_inference_input_2.path}",
        f"--input=1xf16=@{sdxl_unet_inference_input_3.path}",
        f"--expected_output=1x4x128x128xf16=@{sdxl_unet_inference_output_0.path}",
    ]


ROCM_COMPILE_FLAGS = [
    "--iree-hal-target-backends=rocm",
    f"--iree-rocm-target-chip={rocm_chip}",
    "--iree-opt-const-eval=false",
    f"--iree-codegen-transform-dialect-library={iree_test_path_extension}/attention_and_matmul_spec.mlir",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-flow-enable-fuse-horizontal-contractions=true",
    "--iree-flow-enable-aggressive-fusion=true",
    "--iree-opt-aggressively-propagate-transposes=true",
    "--iree-opt-outer-dim-concat=true",
    "--iree-vm-target-truncate-unsupported-floats",
    "--iree-llvmgpu-enable-prefetch=true",
    "--iree-opt-data-tiling=false",
    "--iree-codegen-gpu-native-math-precision=true",
    "--iree-codegen-llvmgpu-use-vector-distribution",
    "--iree-rocm-waves-per-eu=2",
    "--iree-execution-model=async-external",
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics)",
    "--iree-scheduling-dump-statistics-format=json",
    "--iree-scheduling-dump-statistics-file=compilation_info.json",
]

ROCM_PIPELINE_COMPILE_FLAGS = [
    "--iree-hal-target-backends=rocm",
    f"--iree-rocm-target-chip={rocm_chip}",
    "--verify=false",
    "--iree-opt-const-eval=false",
]

###############################################################################
# CPU
###############################################################################


def test_compile_unet_pipeline_cpu(sdxl_unet_pipeline_mlir):
    VmfbManager.sdxl_unet_cpu_pipeline_vmfb = iree_compile(
        sdxl_unet_pipeline_mlir,
        "cpu",
        CPU_COMPILE_FLAGS,
    )


def test_compile_unet_cpu(sdxl_unet_mlir):
    VmfbManager.sdxl_unet_cpu_vmfb = iree_compile(
        sdxl_unet_mlir, "cpu", CPU_COMPILE_FLAGS
    )


@pytest.mark.depends(on=["test_compile_unet_pipeline_cpu", "test_compile_unet_cpu"])
def test_run_unet_cpu(SDXL_UNET_COMMON_RUN_FLAGS, sdxl_unet_real_weights):
    return iree_run_module(
        VmfbManager.sdxl_unet_cpu_vmfb,
        device="local-task",
        function="produce_image_latents",
        args=[
            f"--parameters=model={sdxl_unet_real_weights.path}",
            f"--module={VmfbManager.sdxl_unet_cpu_pipeline_vmfb.path}",
            "--expected_f16_threshold=0.8f",
        ]
        + SDXL_UNET_COMMON_RUN_FLAGS,
    )


###############################################################################
# ROCM
###############################################################################


def test_compile_unet_pipeline_rocm(sdxl_unet_pipeline_mlir):
    VmfbManager.sdxl_unet_rocm_pipeline_vmfb = iree_compile(
        sdxl_unet_pipeline_mlir,
        f"rocm_{rocm_chip}",
        ROCM_PIPELINE_COMPILE_FLAGS,
    )


def test_compile_unet_rocm(sdxl_unet_mlir):
    VmfbManager.sdxl_unet_rocm_vmfb = iree_compile(
        sdxl_unet_mlir, f"rocm_{rocm_chip}", ROCM_COMPILE_FLAGS
    )


@pytest.mark.depends(on=["test_compile_unet_pipeline_rocm", "test_compile_unet_rocm"])
def test_run_unet_rocm(SDXL_UNET_COMMON_RUN_FLAGS, sdxl_unet_real_weights):
    return iree_run_module(
        VmfbManager.sdxl_unet_rocm_vmfb,
        device="hip",
        function="produce_image_latents",
        args=[
            f"--parameters=model={sdxl_unet_real_weights.path}",
            f"--module={VmfbManager.sdxl_unet_rocm_pipeline_vmfb.path}",
            "--expected_f16_threshold=0.7f",
        ]
        + SDXL_UNET_COMMON_RUN_FLAGS,
    )
