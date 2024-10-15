# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers_tools import *
import os
from pathlib import Path
from conftest import VmfbManager

rocm_chip = os.getenv("ROCM_CHIP", default="gfx90a")
vmfb_dir = os.getenv("TEST_OUTPUT_ARTIFACTS", default=Path.cwd())

###############################################################################
# Fixtures
###############################################################################

sd3_mmdit_inference_input_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-mmdit/inference_input.0.bin",
    group="sd3_mmdit",
)

sd3_mmdit_inference_input_1 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-mmdit/inference_input.1.bin",
    group="sd3_mmdit",
)

sd3_mmdit_inference_input_2 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-mmdit/inference_input.2.bin",
    group="sd3_mmdit",
)

sd3_mmdit_inference_input_3 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-mmdit/inference_input.3.bin",
    group="sd3_mmdit",
)

sd3_mmdit_inference_output_0 = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-mmdit/inference_output.0.bin",
    group="sd3_mmdit",
)

sd3_mmdit_real_weights = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-mmdit/real_weights.irpa",
    group="sd3_mmdit",
)

sd3_mmdit_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sd3-mmdit/model.mlirbc",
    group="sd3_mmdit",
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
def SD3_MMDIT_COMMON_RUN_FLAGS(
    sd3_mmdit_inference_input_0,
    sd3_mmdit_inference_input_1,
    sd3_mmdit_inference_input_2,
    sd3_mmdit_inference_input_3,
    sd3_mmdit_inference_output_0,
):
    return [
        f"--input=2x16x128x128xf16=@{sd3_mmdit_inference_input_0.path}",
        f"--input=2x154x4096xf16=@{sd3_mmdit_inference_input_1.path}",
        f"--input=2x2048xf16=@{sd3_mmdit_inference_input_2.path}",
        f"--input=2xf16=@{sd3_mmdit_inference_input_3.path}",
        f"--expected_output=2x16x128x128xf32=@{sd3_mmdit_inference_output_0.path}",
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
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics)",
]

###############################################################################
# CPU
###############################################################################


def test_compile_mmdit_cpu(sd3_mmdit_mlir):
    VmfbManager.sd3_mmdit_cpu_vmfb = iree_compile(
        sd3_mmdit_mlir,
        CPU_COMPILE_FLAGS,
        Path(vmfb_dir)
        / Path("sd3_mmdit_vmfbs")
        / Path(sd3_mmdit_mlir.path.name).with_suffix(f".cpu.vmfb"),
    )


@pytest.mark.xfail(
    strict=True,
    reason="Expected run to fail",
)
@pytest.mark.depends(on=["test_compile_mmdit_cpu"])
def test_run_mmdit_cpu(SD3_MMDIT_COMMON_RUN_FLAGS, sd3_mmdit_real_weights):
    return iree_run_module(
        VmfbManager.sd3_mmdit_cpu_vmfb,
        device="local-task",
        function="run_forward",
        args=[f"--parameters=model={sd3_mmdit_real_weights.path}"]
        + SD3_MMDIT_COMMON_RUN_FLAGS,
    )


###############################################################################
# ROCM
###############################################################################


@pytest.mark.xfail(
    strict=True,
    reason="Expected compilation to fail",
)
def test_compile_mmdit_rocm(sd3_mmdit_mlir):
    VmfbManager.sd3_mmdit_rocm_vmfb = iree_compile(
        sd3_mmdit_mlir,
        ROCM_COMPILE_FLAGS,
        Path(vmfb_dir)
        / Path("sd3_mmdit_vmfbs")
        / Path(sd3_mmdit_mlir.path.name).with_suffix(f".rocm_{rocm_chip}.vmfb"),
    )


@pytest.mark.depends(on=["test_compile_mmdit_rocm"])
def test_run_mmdit_rocm(SD3_MMDIT_COMMON_RUN_FLAGS, sd3_mmdit_real_weights):
    return iree_run_module(
        VmfbManager.sd3_mmdit_rocm_vmfb,
        device="hip",
        function="run_forward",
        args=[f"--parameters=model={sd3_mmdit_real_weights.path}"]
        + SD3_MMDIT_COMMON_RUN_FLAGS,
    )
