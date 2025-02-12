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
rocm_chip = os.getenv("ROCM_CHIP", default="gfx942")
sku = os.getenv("SKU", default="mi300")
iree_test_path_extension = os.getenv("IREE_TEST_PATH_EXTENSION", default=Path.cwd())

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
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/model.mlir",
    group="sdxl_unet_fp16",
)

sdxl_unet_fp16_pipeline_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduled-unet/sdxl_unet_pipeline_bench_f16.mlir",
    group="sdxl_unet_fp16",
)

# FP16 Model for 960x1024 image size

sdxl_unet_fp16_960_1024_inference_input_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/ian/unet_npys/input1.npy",
    group="sdxl_unet_fp16_960_1024",
)

sdxl_unet_fp16_960_1024_inference_input_1 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/ian/unet_npys/input2.npy",
    group="sdxl_unet_fp16_960_1024",
)

sdxl_unet_fp16_960_1024_inference_input_2 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/ian/unet_npys/input3.npy",
    group="sdxl_unet_fp16_960_1024",
)

sdxl_unet_fp16_960_1024_inference_input_3 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/ian/unet_npys/input4.npy",
    group="sdxl_unet_fp16_960_1024",
)

sdxl_unet_fp16_960_1024_inference_input_4 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/ian/unet_npys/input5.npy",
    group="sdxl_unet_fp16_960_1024",
)

sdxl_unet_fp16_960_1024_inference_input_5 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/ian/unet_npys/input6.npy",
    group="sdxl_unet_fp16_960_1024",
)

sdxl_unet_fp16_960_1024_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/ian/unet_npys/golden_out.npy",
    group="sdxl_unet_fp16_960_1024",
)

sdxl_unet_fp16_960_1024_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/ian/sdxl_960x1024/stable_diffusion_xl_base_1_0_bs1_64_960x1024_fp16_unet.mlir",
    group="sdxl_unet_fp16_960_1024",
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
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/11-13-2024/punet_fp16_out.0.bin",
    group="sdxl_punet_int8_fp16",
)

sdxl_punet_int8_fp16_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/punet_weights.irpa",
    group="sdxl_punet_int8_fp16",
)

sdxl_punet_int8_fp16_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/11-8-2024/punet_fp16.mlir",
    group="sdxl_punet_int8_fp16",
)

# INT8 Punet + FP8 Attention

sdxl_punet_int8_fp8_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/11-13-2024/punet_fp8_out.0.bin",
    group="sdxl_punet_int8_fp8",
)

sdxl_punet_int8_fp8_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/punet_fp8_weights.irpa",
    group="sdxl_punet_int8_fp8",
)

sdxl_punet_int8_fp8_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/stan/sdxl-punet/11-26-2024/punet_fp8.mlir",
    group="sdxl_punet_int8_fp8",
)

CPU_COMPILE_FLAGS = [
    "--iree-hal-target-backends=llvm-cpu",
    "--iree-llvmcpu-target-cpu-features=host",
    "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false",
    "--iree-llvmcpu-distribution-size=32",
    "--iree-opt-const-eval=false",
    "--iree-opt-strip-assertions=true",
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
def SDXL_UNET_FP16_960_1024_COMMON_RUN_FLAGS(
    sdxl_unet_fp16_960_1024_inference_input_0,
    sdxl_unet_fp16_960_1024_inference_input_1,
    sdxl_unet_fp16_960_1024_inference_input_2,
    sdxl_unet_fp16_960_1024_inference_input_3,
    sdxl_unet_fp16_960_1024_inference_input_4,
    sdxl_unet_fp16_960_1024_inference_input_5,
    sdxl_unet_fp16_960_1024_inference_output_0,
):
    return [
        f"--input=@{sdxl_unet_fp16_960_1024_inference_input_0.path}",
        f"--input=@{sdxl_unet_fp16_960_1024_inference_input_1.path}",
        f"--input=@{sdxl_unet_fp16_960_1024_inference_input_2.path}",
        f"--input=@{sdxl_unet_fp16_960_1024_inference_input_3.path}",
        f"--input=@{sdxl_unet_fp16_960_1024_inference_input_4.path}",
        f"--input=@{sdxl_unet_fp16_960_1024_inference_input_5.path}",
        f"--expected_output=@{sdxl_unet_fp16_960_1024_inference_output_0.path}",
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
    "--iree-opt-strip-assertions=true",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-dispatch-creation-enable-fuse-horizontal-contractions=true",
    "--iree-dispatch-creation-enable-aggressive-fusion=true",
    "--iree-opt-aggressively-propagate-transposes=true",
    "--iree-opt-outer-dim-concat=true",
    "--iree-opt-generalize-matmul=true",
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
if os.path.isfile(
    f"{iree_test_path_extension}/attention_and_matmul_spec_unet_fp16_{sku}.mlir"
):
    FP16_UNET_FLAGS.append(
        f"--iree-codegen-transform-dialect-library={iree_test_path_extension}/attention_and_matmul_spec_unet_fp16_{sku}.mlir"
    )

INT8_PUNET_FLAGS = [
    "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics)",
]

if os.path.isfile(
    f"{iree_test_path_extension}/attention_and_matmul_spec_punet_{sku}.mlir"
):
    INT8_PUNET_FLAGS.append(
        f"--iree-codegen-transform-dialect-library={iree_test_path_extension}/attention_and_matmul_spec_punet_{sku}.mlir"
    )
else:
    # TODO: Investigate numerics failure without using the MI300 punet attention spec
    INT8_PUNET_FLAGS.append(
        f"--iree-codegen-transform-dialect-library={iree_test_path_extension}/attention_and_matmul_spec_punet_mi300.mlir"
    )


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


def test_compile_unet_fp16_960_1024_cpu(sdxl_unet_fp16_960_1024_mlir):
    VmfbManager.sdxl_unet_fp16_960_1024_cpu_vfmb = iree_compile(
        sdxl_unet_fp16_960_1024_mlir,
        CPU_COMPILE_FLAGS,
        Path(vmfb_dir)
        / Path("sdxl_unet_fp16_960_1024_vmfbs")
        / Path(sdxl_unet_fp16_960_1024_mlir.path.name).with_suffix(f".cpu.vmfb"),
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


@pytest.mark.depends(on=["test_compile_unet_fp16_cpu"])
def test_run_unet_fp16_960_1024_cpu(
    SDXL_UNET_FP16_960_1024_COMMON_RUN_FLAGS, sdxl_unet_fp16_real_weights
):
    return iree_run_module(
        VmfbManager.sdxl_unet_fp16_960_1024_cpu_vfmb,
        device="local-task",
        function="run_forward",
        args=[
            f"--parameters=model={sdxl_unet_fp16_real_weights.path}",
            f"--module={VmfbManager.sdxl_unet_fp16_960_1024_cpu_vfmb}",
            "--expected_f16_threshold=0.8f",
        ]
        + SDXL_UNET_FP16_960_1024_COMMON_RUN_FLAGS,
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


def test_compile_unet_fp16_960_1024_rocm(sdxl_unet_fp16_960_1024_mlir):
    VmfbManager.sdxl_unet_fp16_960_1024_rocm_vmfb = iree_compile(
        sdxl_unet_fp16_960_1024_mlir,
        ROCM_COMPILE_FLAGS + FP16_UNET_FLAGS,
        Path(vmfb_dir)
        / Path("sdxl_unet_fp16_960_1024_vmfbs")
        / Path(sdxl_unet_fp16_960_1024_mlir.path.name).with_suffix(
            f".rocm_{rocm_chip}.vmfb"
        ),
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


@pytest.mark.depends(on=["test_compile_unet_fp16_960_1024_rocm"])
def test_run_unet_fp16_rocm(
    SDXL_UNET_FP16_960_1024_COMMON_RUN_FLAGS, sdxl_unet_fp16_real_weights
):
    return iree_run_module(
        VmfbManager.sdxl_unet_fp16_960_1024_rocm_vmfb,
        device="hip",
        function="run_forward",
        args=[
            f"--parameters=model={sdxl_unet_fp16_real_weights.path}",
            f"--module={VmfbManager.sdxl_unet_fp16_960_1024_rocm_vmfb}",
            "--expected_f16_threshold=0.705f",
        ]
        + SDXL_UNET_FP16_960_1024_COMMON_RUN_FLAGS,
    )


def test_compile_punet_int8_fp16_rocm(request, sdxl_punet_int8_fp16_mlir):
    if rocm_chip == "gfx90a":
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Expected punet_int8_fp8 compilation on mi250 to fail",
                strict=True,
            )
        )
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
