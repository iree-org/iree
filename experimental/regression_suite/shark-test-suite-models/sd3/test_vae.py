# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers import *
import os

rocm_chip = os.getenv("ROCM_CHIP", default="gfx90a")

###############################################################################
# Fixtures
###############################################################################

sd3_vae_input_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-vae/inference_input.0.bin",
    group="sd3_vae_input_0",
)

sd3_vae_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-vae/inference_output.0.bin",
    group="sd3_vae_output_0",
)

sd3_vae_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-vae/real_weights.irpa",
    group="sd3_vae_real_weights",
)

sd3_vae_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-vae/model.mlirbc",
    group="sd3_vae_mlir",
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

COMMON_RUN_FLAGS = [
    f"--input=1x16x128x128xf16=@{sd3_vae_inference_input_0.path}",
    f"--expected_output=3x1024x1024xf32=@{sd3_vae_inference_output_0.path}",
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
]

###############################################################################
# CPU
###############################################################################

cpu_vmfb = None


def test_compile_vae_cpu():
    cpu_vmfb = iree_compile(sd3_vae_mlir, "cpu", CPU_COMPILE_FLAGS)


@pytest.mark.depends(on=["test_compile_vae_cpu"])
def test_run_vae_cpu():
    return iree_run_module(
        cpu_vmfb,
        device="local-task",
        function="decode",
        args = [
            f"--parameters=model={sd3_vae_real_weights.path}",
            "--expected_f32_threshold=0.01f",
        ]
        + COMMON_RUN_FLAGS,
    )


###############################################################################
# ROCM
###############################################################################

rocm_vmfb = None

def test_compile_vae_rocm():
    rocm_vmfb = iree_compile(sd3_vae_mlir, f"rocm_{rocm_chip}", ROCM_COMPILE_FLAGS)


@pytest.mark.depends(on=["test_compile_vae_rocm"])
def test_run_vae_rocm():
    return iree_run_module(
        rocm_vmfb,
        device="hip",
        function="decode",
        args = [
            f"--parameters=model={sd3_vae_real_weights.path}",
            "--expected_f32_threshold=0.7f",
        ]
        + COMMON_RUN_FLAGS,
    )
