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

sd3_clip_inference_input_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.0.bin",
    group="sd3_clip_inference_input_0",
)

sd3_clip_inference_input_1 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.1.bin",
    group="sd3_clip_inference_input_1",
)

sd3_clip_inference_input_2 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.2.bin",
    group="sd3_clip_inference_input_2",
)

sd3_clip_inference_input_3 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.3.bin",
    group="sd3_clip_inference_input_3",
)

sd3_clip_inference_input_4 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.4.bin",
    group="sd3_clip_inference_input_4",
)

sd3_clip_inference_input_5 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.5.bin",
    group="sd3_clip_inference_input_5",
)

sd3_clip_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_output.0.bin",
    group="sd3_clip_inference_output_0",
)

sd3_clip_inference_output_1 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_output.1.bin",
    group="sd3_clip_inference_output_1",
)

sd3_clip_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/real_weights.irpa",
    group="sd3_clip_real_weights",
)

sd3_clip_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/model.mlirbc",
    group="sd3_clip_mlir",
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
    f"--input=1x77x2xi64=@{sd3_clip_inference_input_0.path}",
    f"--input=1x77x2xi64=@{sd3_clip_inference_input_1.path}",
    f"--input=1x77x2xi64=@{sd3_clip_inference_input_2.path}",
    f"--input=1x77x2xi64=@{sd3_clip_inference_input_3.path}",
    f"--input=1x77x2xi64=@{sd3_clip_inference_input_4.path}",
    f"--input=1x77x2xi64=@{sd3_clip_inference_input_5.path}",
    f"--expected_output=2x154x4096xf32=@{sd3_clip_inference_output_0.path}",
    f"--expected_output=2x2048xf32=@{sd3_clip_inference_output_1.path}",
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
]

###############################################################################
# CPU
###############################################################################

cpu_vmfb = None

def test_compile_clip_cpu():
    cpu_vmfb = iree_compile(sd3_clip_mlir, "cpu", CPU_COMPILE_FLAGS)


@pytest.mark.depends(on=["test_compile_clip_cpu"])
def test_run_clip_cpu():
    iree_run_module(
        cpu_vmfb,
        device="local-task",
        function="encode_tokens",
        args = [
            f"--parameters=model={sd3_clip_real_weights.path}",
            "--expected_f32_threshold=0.15f",
        ]
        + COMMON_RUN_FLAGS
    )


###############################################################################
# ROCM
###############################################################################

rocm_vmfb = None

@pytest.mark.xfail(
    raises=IreeCompileException,
    strict=True,
    reason="Expected compilation to fail (remove xfail for test_compile_clip_rocm)",
)
def test_compile_clip_rocm():
    rocm_vmfb = iree_compile(sd3_clip_mlir, f"rocm_{rocm_chip}", ROCM_COMPILE_FLAGS)


@pytest.mark.depends(on=["test_compile_clip_rocm"])
def test_run_clip_rocm():
    return iree_run_module(
        rocm_vmfb,
        device="hip",
        function="encode_tokens",
        args=[f"--parameters=model={sd3_clip_real_weights.path}"] + COMMON_RUN_FLAGS,
    )
