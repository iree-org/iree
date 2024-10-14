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

vmfb_dir = os.getenv("TEST_OUTPUT_ARTIFACTS", default=Path.cwd())
rocm_chip = os.getenv("ROCM_CHIP", default="gfx90a")

###############################################################################
# Fixtures
###############################################################################

# FP16 Model

sdxl_unet_fp16_inference_input_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/inference_input.0.bin",
    group="sdxl_unet_fp16",
)

sdxl_unet_fp16_inference_input_1 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/inference_input.1.bin",
    group="sdxl_unet_fp16",
)

sdxl_unet_fp16_inference_input_2 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/inference_input.2.bin",
    group="sdxl_unet_fp16",
)

sdxl_unet_fp16_inference_input_3 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/inference_input.3.bin",
    group="sdxl_unet_fp16",
)

sdxl_unet_fp16_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/inference_output.0.bin",
    group="sdxl_unet_fp16",
)

sdxl_unet_fp16_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/real_weights.irpa",
    group="sdxl_unet_fp16",
)

sdxl_unet_fp16_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/model.mlirbc",
    group="sdxl_unet_fp16",
)

sdxl_unet_fp16_pipeline_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/sdxl_unet_pipeline_bench_f16.mlir",
    group="sdxl_unet_fp16",
)

# INT8 Punet + FP16 Attention

sdxl_punet_int8_inference_input_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.0.bin",
    group="sdxl_punet_int8",
)

sdxl_punet_int8_inference_input_1 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.1.bin",
    group="sdxl_punet_int8",
)

sdxl_punet_int8_inference_input_2 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.2.bin",
    group="sdxl_punet_int8",
)

sdxl_punet_int8_inference_input_3 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.3.bin",
    group="sdxl_punet_int8",
)

sdxl_punet_int8_inference_input_4 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.4.bin",
    group="sdxl_punet_int8",
)

sdxl_punet_int8_inference_input_5 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.5.bin",
    group="sdxl_punet_int8",
)

sdxl_punet_int8_fp16_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/new_punet_out.0.bin",
    group="sdxl_punet_int8_fp16",
)

sdxl_punet_int8_fp16_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/punet_weights.irpa",
    group="sdxl_punet_int8_fp16",
)

sdxl_punet_int8_fp16_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/punet.mlir",
    group="sdxl_punet_int8_fp16",
)

# INT8 Punet + FP8 Attention

sdxl_punet_int8_fp8_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/new_punet_fp8_out.0.bin",
    group="sdxl_punet_int8_fp8",
)

sdxl_punet_int8_fp8_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/punet_fp8_weights.irpa",
    group="sdxl_punet_int8_fp8",
)

sdxl_punet_int8_fp8_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/rob/sdxl-punet/punet_fp8.mlir",
    group="sdxl_punet_int8_fp8",
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
def SDXL_UNET_FP16_COMMON_RUN_FLAGS(
    sdxl_unet_fp16_inference_input_0,
    sdxl_unet_fp16_inference_input_1,
    sdxl_unet_fp16_inference_input_2,
    sdxl_unet_fp16_inference_input_3,
    sdxl_unet_fp16_inference_output_0,
):
    return [
        f"--input=1x4x128x128xf16=@{sdxl_unet_fp16_inference_input_0.path}",
        f"--input=2x64x2048xf16=@{sdxl_unet_fp16_inference_input_1.path}",
        f"--input=2x1280xf16=@{sdxl_unet_fp16_inference_input_2.path}",
        f"--input=1xf16=@{sdxl_unet_fp16_inference_input_3.path}",
        f"--expected_output=1x4x128x128xf16=@{sdxl_unet_fp16_inference_output_0.path}",
    ]


@pytest.fixture
def SDXL_PUNET_INT8_COMMON_RUN_FLAGS(
    sdxl_punet_int8_inference_input_0,
    sdxl_punet_int8_inference_input_1,
    sdxl_punet_int8_inference_input_2,
    sdxl_punet_int8_inference_input_3,
    sdxl_punet_int8_inference_input_4,
    sdxl_punet_int8_inference_input_5,
):
    return [
        f"--input=1x4x128x128xf16=@{sdxl_punet_int8_inference_input_0.path}",
        f"--input=1xf16=@{sdxl_punet_int8_inference_input_1.path}",
        f"--input=2x64x2048xf16=@{sdxl_punet_int8_inference_input_2.path}",
        f"--input=2x1280xf16=@{sdxl_punet_int8_inference_input_3.path}",
        f"--input=2x6xf16=@{sdxl_punet_int8_inference_input_4.path}",
        f"--input=1xf16=@{sdxl_punet_int8_inference_input_5.path}",
    ]


@pytest.fixture
def SDXL_PUNET_INT8_FP16_OUT(
    sdxl_punet_int8_fp16_inference_output_0,
):
    return [
        f"--expected_output=1x4x128x128xf16=@{sdxl_punet_int8_fp16_inference_output_0.path}",
    ]


@pytest.fixture
def SDXL_PUNET_INT8_FP8_OUT(
    sdxl_punet_int8_fp8_inference_output_0,
):
    return [
        f"--expected_output=1x4x128x128xf16=@{sdxl_punet_int8_fp8_inference_output_0.path}",
    ]


ROCM_COMPILE_FLAGS = [
    "--iree-hal-target-backends=rocm",
    f"--iree-hip-target={rocm_chip}",
    "--iree-opt-const-eval=false",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-dispatch-creation-enable-fuse-horizontal-contractions=true",
    "--iree-dispatch-creation-enable-aggressive-fusion=true",
    "--iree-opt-aggressively-propagate-transposes=true",
    "--iree-opt-outer-dim-concat=true",
    "--iree-vm-target-truncate-unsupported-floats",
    "--iree-llvmgpu-enable-prefetch=true",
    "--iree-opt-data-tiling=false",
    "--iree-codegen-gpu-native-math-precision=true",
    "--iree-codegen-llvmgpu-use-vector-distribution",
    "--iree-hip-waves-per-eu=2",
    "--iree-execution-model=async-external",
    "--iree-scheduling-dump-statistics-format=json",
    "--iree-scheduling-dump-statistics-file=compilation_info.json",
]

FP16_UNET_FLAGS = [
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics)",
]

INT8_PUNET_FLAGS = [
    "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))",
]

ROCM_UNET_PIPELINE_FP16_COMPILE_FLAGS = [
    "--iree-hal-target-backends=rocm",
    f"--iree-hip-target={rocm_chip}",
    "--verify=false",
    "--iree-opt-const-eval=false",
]

###############################################################################
# CPU
###############################################################################


def test_compile_unet_fp16_pipeline_cpu(sdxl_unet_fp16_pipeline_mlir):
    VmfbManager.sdxl_unet_fp16_cpu_pipeline_vmfb = iree_compile(
        sdxl_unet_fp16_pipeline_mlir,
        CPU_COMPILE_FLAGS,
        Path(vmfb_dir)
        / Path("sdxl_unet_fp16_vmfbs")
        / Path(sdxl_unet_fp16_pipeline_mlir.path.name).with_suffix(f".cpu.vmfb"),
    )


def test_compile_unet_fp16_cpu(sdxl_unet_fp16_mlir):
    VmfbManager.sdxl_unet_fp16_cpu_vmfb = iree_compile(
        sdxl_unet_fp16_mlir,
        CPU_COMPILE_FLAGS,
        Path(vmfb_dir)
        / Path("sdxl_unet_fp16_vmfbs")
        / Path(sdxl_unet_fp16_mlir.path.name).with_suffix(f".cpu.vmfb"),
    )


@pytest.mark.depends(
    on=["test_compile_unet_fp16_pipeline_cpu", "test_compile_unet_fp16_cpu"]
)
def test_run_unet_fp16_cpu(
    SDXL_UNET_FP16_COMMON_RUN_FLAGS, sdxl_unet_fp16_real_weights
):
    return iree_run_module(
        VmfbManager.sdxl_unet_fp16_cpu_vmfb,
        device="local-task",
        function="produce_image_latents",
        args=[
            f"--parameters=model={sdxl_unet_fp16_real_weights.path}",
            f"--module={VmfbManager.sdxl_unet_fp16_cpu_pipeline_vmfb}",
            "--expected_f16_threshold=0.8f",
        ]
        + SDXL_UNET_FP16_COMMON_RUN_FLAGS,
    )


###############################################################################
# ROCM
###############################################################################


def test_compile_unet_fp16_pipeline_rocm(sdxl_unet_fp16_pipeline_mlir):
    VmfbManager.sdxl_unet_fp16_rocm_pipeline_vmfb = iree_compile(
        sdxl_unet_fp16_pipeline_mlir,
        ROCM_UNET_PIPELINE_FP16_COMPILE_FLAGS,
        Path(vmfb_dir)
        / Path("sdxl_unet_fp16_vmfbs")
        / Path(sdxl_unet_fp16_pipeline_mlir.path.name).with_suffix(
            f".rocm_{rocm_chip}.vmfb"
        ),
    )


def test_compile_unet_fp16_rocm(sdxl_unet_fp16_mlir):
    VmfbManager.sdxl_unet_fp16_rocm_vmfb = iree_compile(
        sdxl_unet_fp16_mlir,
        ROCM_COMPILE_FLAGS + FP16_UNET_FLAGS,
        Path(vmfb_dir)
        / Path("sdxl_unet_fp16_vmfbs")
        / Path(sdxl_unet_fp16_mlir.path.name).with_suffix(f".rocm_{rocm_chip}.vmfb"),
    )


@pytest.mark.depends(
    on=["test_compile_unet_fp16_pipeline_rocm", "test_compile_unet_fp16_rocm"]
)
def test_run_unet_fp16_rocm(
    SDXL_UNET_FP16_COMMON_RUN_FLAGS, sdxl_unet_fp16_real_weights
):
    return iree_run_module(
        VmfbManager.sdxl_unet_fp16_rocm_vmfb,
        device="hip",
        function="produce_image_latents",
        args=[
            f"--parameters=model={sdxl_unet_fp16_real_weights.path}",
            f"--module={VmfbManager.sdxl_unet_fp16_rocm_pipeline_vmfb}",
            "--expected_f16_threshold=0.705f",
        ]
        + SDXL_UNET_FP16_COMMON_RUN_FLAGS,
    )


def test_compile_punet_int8_fp16_rocm(sdxl_punet_int8_fp16_mlir):
    VmfbManager.sdxl_punet_int8_fp16_rocm_vmfb = iree_compile(
        sdxl_punet_int8_fp16_mlir,
        ROCM_COMPILE_FLAGS + INT8_PUNET_FLAGS,
        Path(vmfb_dir)
        / Path("sdxl_punet_int8_fp16_vmfbs")
        / Path(sdxl_punet_int8_fp16_mlir.path.name).with_suffix(
            f".rocm_{rocm_chip}.vmfb"
        ),
    )


@pytest.mark.depends(on=["test_compile_punet_int8_fp16_rocm"])
def test_run_punet_int8_fp16_rocm(
    request,
    SDXL_PUNET_INT8_COMMON_RUN_FLAGS,
    SDXL_PUNET_INT8_FP16_OUT,
    sdxl_punet_int8_fp16_real_weights,
):
    if rocm_chip == "gfx90a":
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Expected punet_int8_fp16 run on mi250 to fail", strict=True
            )
        )
    return iree_run_module(
        VmfbManager.sdxl_punet_int8_fp16_rocm_vmfb,
        device="hip",
        function="main",
        args=[
            f"--parameters=model={sdxl_punet_int8_fp16_real_weights.path}",
        ]
        + SDXL_PUNET_INT8_COMMON_RUN_FLAGS
        + SDXL_PUNET_INT8_FP16_OUT,
    )


def test_compile_punet_int8_fp8_rocm(request, sdxl_punet_int8_fp8_mlir):
    if rocm_chip == "gfx90a":
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Expected punet_int8_fp8 compilation on mi250 to fail",
                strict=True,
            )
        )
    VmfbManager.sdxl_punet_int8_fp8_rocm_vmfb = iree_compile(
        sdxl_punet_int8_fp8_mlir,
        ROCM_COMPILE_FLAGS + INT8_PUNET_FLAGS,
        Path(vmfb_dir)
        / Path("sdxl_punet_int8_fp8_vmfbs")
        / Path(sdxl_punet_int8_fp8_mlir.path.name).with_suffix(
            f".rocm_{rocm_chip}.vmfb"
        ),
    )


@pytest.mark.depends(on=["test_compile_punet_int8_fp8_rocm"])
def test_run_punet_int8_fp8_rocm(
    SDXL_PUNET_INT8_COMMON_RUN_FLAGS,
    SDXL_PUNET_INT8_FP8_OUT,
    sdxl_punet_int8_fp8_real_weights,
):
    return iree_run_module(
        VmfbManager.sdxl_punet_int8_fp8_rocm_vmfb,
        device="hip",
        function="main",
        args=[
            f"--parameters=model={sdxl_punet_int8_fp8_real_weights.path}",
        ]
        + SDXL_PUNET_INT8_COMMON_RUN_FLAGS
        + SDXL_PUNET_INT8_FP8_OUT,
    )
