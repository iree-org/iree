# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers_tools import *
import os
from conftest import VmfbManager
from pathlib import Path

rocm_chip = os.getenv("ROCM_CHIP", default="gfx90a")
vmfb_dir = os.getenv("TEST_OUTPUT_ARTIFACTS", default=Path.cwd())

###############################################################################
# Fixtures
###############################################################################

sd3_clip_inference_input_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.0.bin",
    group="sd3_clip",
)

sd3_clip_inference_input_1 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.1.bin",
    group="sd3_clip",
)

sd3_clip_inference_input_2 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.2.bin",
    group="sd3_clip",
)

sd3_clip_inference_input_3 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.3.bin",
    group="sd3_clip",
)

sd3_clip_inference_input_4 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.4.bin",
    group="sd3_clip",
)

sd3_clip_inference_input_5 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_input.5.bin",
    group="sd3_clip",
)

sd3_clip_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_output.0.bin",
    group="sd3_clip",
)

sd3_clip_inference_output_1 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/inference_output.1.bin",
    group="sd3_clip",
)

sd3_clip_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/real_weights.irpa",
    group="sd3_clip",
)

sd3_clip_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-prompt-encoder/model.mlirbc",
    group="sd3_clip",
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
def SD3_CLIP_COMMON_RUN_FLAGS(
    sd3_clip_inference_input_0,
    sd3_clip_inference_input_1,
    sd3_clip_inference_input_2,
    sd3_clip_inference_input_3,
    sd3_clip_inference_input_4,
    sd3_clip_inference_input_5,
    sd3_clip_inference_output_0,
    sd3_clip_inference_output_1,
):
    return [
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
    f"--iree-hip-target={rocm_chip}",
    "--iree-input-type=torch",
    "--iree-opt-const-eval=false",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-opt-outer-dim-concat=true",
    "--iree-hip-waves-per-eu=2",
    "--iree-llvmgpu-enable-prefetch",
    "--iree-dispatch-creation-enable-aggressive-fusion",
    "--iree-dispatch-creation-enable-fuse-horizontal-contractions=true",
    "--iree-opt-aggressively-propagate-transposes=true",
    "--iree-codegen-llvmgpu-use-vector-distribution=true",
    "--iree-execution-model=async-external",
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics{pad-target-type=conv})",
]

###############################################################################
# CPU
###############################################################################


def test_compile_clip_cpu(sd3_clip_mlir):
    VmfbManager.sd3_clip_cpu_vmfb = iree_compile(
        sd3_clip_mlir,
        CPU_COMPILE_FLAGS,
        Path(vmfb_dir)
        / Path("sd3_clip_vmfbs")
        / Path(sd3_clip_mlir.path.name).with_suffix(f".cpu.vmfb"),
    )


@pytest.mark.depends(on=["test_compile_clip_cpu"])
def test_run_clip_cpu(SD3_CLIP_COMMON_RUN_FLAGS, sd3_clip_real_weights):
    iree_run_module(
        VmfbManager.sd3_clip_cpu_vmfb,
        device="local-task",
        function="encode_tokens",
        args=[
            f"--parameters=model={sd3_clip_real_weights.path}",
            "--expected_f32_threshold=0.15f",
        ]
        + SD3_CLIP_COMMON_RUN_FLAGS,
    )


###############################################################################
# ROCM
###############################################################################


@pytest.mark.xfail(
    strict=True,
    reason="Expected compilation to fail",
)
def test_compile_clip_rocm(sd3_clip_mlir):
    VmfbManager.sd3_clip_rocm_vmfb = iree_compile(
        sd3_clip_mlir,
        ROCM_COMPILE_FLAGS,
        Path(vmfb_dir)
        / Path("sd3_clip_vmfbs")
        / Path(sd3_clip_mlir.path.name).with_suffix(f".rocm_{rocm_chip}.vmfb"),
    )


@pytest.mark.depends(on=["test_compile_clip_rocm"])
def test_run_clip_rocm(SD3_CLIP_COMMON_RUN_FLAGS, sd3_clip_real_weights):
    return iree_run_module(
        VmfbManager.sd3_clip_rocm_vmfb,
        device="hip",
        function="encode_tokens",
        args=[f"--parameters=model={sd3_clip_real_weights.path}"]
        + SD3_CLIP_COMMON_RUN_FLAGS,
    )
