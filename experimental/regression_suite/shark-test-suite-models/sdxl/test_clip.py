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

sdxl_clip_inference_input_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-prompt-encoder/inference_input.0.bin",
    group="sdxl_clip",
)

sdxl_clip_inference_input_1 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-prompt-encoder/inference_input.1.bin",
    group="sdxl_clip",
)

sdxl_clip_inference_input_2 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-prompt-encoder/inference_input.2.bin",
    group="sdxl_clip",
)

sdxl_clip_inference_input_3 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-prompt-encoder/inference_input.3.bin",
    group="sdxl_clip",
)

sdxl_clip_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-prompt-encoder/inference_output.0.bin",
    group="sdxl_clip",
)

sdxl_clip_inference_output_1 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-prompt-encoder/inference_output.1.bin",
    group="sdxl_clip",
)

sdxl_clip_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-prompt-encoder/real_weights.irpa",
    group="sdxl_clip",
)

sdxl_clip_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-prompt-encoder/model.mlirbc",
    group="sdxl_clip",
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
def SDXL_CLIP_COMMON_RUN_FLAGS(
    sdxl_clip_inference_input_0,
    sdxl_clip_inference_input_1,
    sdxl_clip_inference_input_2,
    sdxl_clip_inference_input_3,
    sdxl_clip_inference_output_0,
    sdxl_clip_inference_output_1,
):
    return [
        f"--input=1x64xi64=@{sdxl_clip_inference_input_0.path}",
        f"--input=1x64xi64=@{sdxl_clip_inference_input_1.path}",
        f"--input=1x64xi64=@{sdxl_clip_inference_input_2.path}",
        f"--input=1x64xi64=@{sdxl_clip_inference_input_3.path}",
        f"--expected_output=2x64x2048xf16=@{sdxl_clip_inference_output_0.path}",
        f"--expected_output=2x1280xf16=@{sdxl_clip_inference_output_1.path}",
    ]


ROCM_COMPILE_FLAGS = [
    "--iree-hal-target-backends=rocm",
    f"--iree-rocm-target-chip={rocm_chip}",
    "--iree-input-type=torch",
    "--iree-opt-const-eval=false",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-opt-outer-dim-concat=true",
    "--iree-rocm-waves-per-eu=2",
    "--iree-llvmgpu-enable-prefetch",
    "--iree-flow-enable-aggressive-fusion",
    "--iree-global-opt-enable-fuse-horizontal-contractions=true",
    "--iree-opt-aggressively-propagate-transposes=true",
    "--iree-codegen-llvmgpu-use-vector-distribution=true",
    "--iree-execution-model=async-external",
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, util.func(iree-preprocessing-pad-to-intrinsics{pad-target-type=conv}))",
    "--iree-scheduling-dump-statistics-format=json",
    "--iree-scheduling-dump-statistics-file=compilation_info.json",
]

###############################################################################
# CPU
###############################################################################


def test_compile_clip_cpu(sdxl_clip_mlir):
    VmfbManager.sdxl_clip_cpu_vmfb = iree_compile(
        sdxl_clip_mlir, "cpu", CPU_COMPILE_FLAGS
    )


@pytest.mark.depends(on=["test_compile_clip_cpu"])
def test_run_clip_cpu(SDXL_CLIP_COMMON_RUN_FLAGS, sdxl_clip_real_weights):
    iree_run_module(
        VmfbManager.sdxl_clip_cpu_vmfb,
        device="local-task",
        function="encode_prompts",
        args=[
            f"--parameters=model={sdxl_clip_real_weights.path}",
            "--expected_f16_threshold=1.0f",
        ]
        + SDXL_CLIP_COMMON_RUN_FLAGS,
    )


###############################################################################
# ROCM
###############################################################################


def test_compile_clip_rocm(sdxl_clip_mlir):
    VmfbManager.sdxl_clip_rocm_vmfb = iree_compile(
        sdxl_clip_mlir, f"rocm_{rocm_chip}", ROCM_COMPILE_FLAGS
    )


@pytest.mark.depends(on=["test_compile_clip_rocm"])
def test_run_clip_rocm(SDXL_CLIP_COMMON_RUN_FLAGS, sdxl_clip_real_weights):
    return iree_run_module(
        VmfbManager.sdxl_clip_rocm_vmfb,
        device="hip",
        function="encode_prompts",
        args=[
            f"--parameters=model={sdxl_clip_real_weights.path}",
            "--expected_f16_threshold=1.0f",
        ]
        + SDXL_CLIP_COMMON_RUN_FLAGS,
    )
