# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers_tools import *
import os
from conftest import VmfbManager

rocm_chip = os.getenv("ROCM_CHIP", default="gfx90a")

###############################################################################
# Fixtures
###############################################################################

sdxl_vae_inference_input_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-vae-decode/inference_input.0.bin",
    group="sdxl_vae",
)

sdxl_vae_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-vae-decode/inference_output.0.bin",
    group="sdxl_vae",
)

sdxl_vae_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-vae-decode/real_weights.irpa",
    group="sdxl_vae",
)

sdxl_vae_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-vae-decode/model.mlirbc",
    group="sdxl_vae",
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
def SDXL_VAE_COMMON_RUN_FLAGS(
    sdxl_vae_inference_input_0,
    sdxl_vae_inference_output_0,
):
    return [
        f"--input=1x4x128x128xf16=@{sdxl_vae_inference_input_0.path}",
        f"--expected_output=1x3x1024x1024xf16=@{sdxl_vae_inference_output_0.path}",
    ]


ROCM_COMPILE_FLAGS = [
    "--iree-hal-target-backends=rocm",
    f"--iree-rocm-target-chip={rocm_chip}",
    "--iree-opt-const-eval=false",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-opt-outer-dim-concat=true",
    "--iree-llvmgpu-enable-prefetch=true",
    "--iree-rocm-waves-per-eu=2",
    "--iree-flow-enable-aggressive-fusion=true",
    "--iree-codegen-llvmgpu-use-vector-distribution=true",
    "--iree-execution-model=async-external",
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, util.func(iree-preprocessing-pad-to-intrinsics))",
    "--iree-scheduling-dump-statistics-format=json",
    "--iree-scheduling-dump-statistics-file=compilation_info.json",
]

###############################################################################
# CPU
###############################################################################


def test_compile_vae_cpu(sdxl_vae_mlir):
    VmfbManager.sdxl_vae_cpu_vmfb = iree_compile(
        sdxl_vae_mlir, "cpu", CPU_COMPILE_FLAGS
    )


@pytest.mark.depends(on=["test_compile_vae_cpu"])
def test_run_vae_cpu(SDXL_VAE_COMMON_RUN_FLAGS, sdxl_vae_real_weights):
    return iree_run_module(
        VmfbManager.sdxl_vae_cpu_vmfb,
        device="local-task",
        function="main",
        args=[
            f"--parameters=model={sdxl_vae_real_weights.path}",
            "--expected_f16_threshold=0.02f",
        ]
        + SDXL_VAE_COMMON_RUN_FLAGS,
    )


###############################################################################
# ROCM
###############################################################################


def test_compile_vae_rocm(sdxl_vae_mlir):
    VmfbManager.sdxl_vae_rocm_vmfb = iree_compile(
        sdxl_vae_mlir, f"rocm_{rocm_chip}", ROCM_COMPILE_FLAGS
    )


@pytest.mark.depends(on=["test_compile_vae_rocm"])
def test_run_vae_rocm(SDXL_VAE_COMMON_RUN_FLAGS, sdxl_vae_real_weights):
    return iree_run_module(
        VmfbManager.sdxl_vae_rocm_vmfb,
        device="hip",
        function="main",
        args=[
            f"--parameters=model={sdxl_vae_real_weights.path}",
            "--expected_f16_threshold=0.4f",
        ]
        + SDXL_VAE_COMMON_RUN_FLAGS,
    )
