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

rocm_chip = os.getenv("ROCM_CHIP", default="gfx942")
vmfb_dir = os.getenv("TEST_OUTPUT_ARTIFACTS", default=Path.cwd())

###############################################################################
# Fixtures
###############################################################################

sdxl_scheduler_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-scheduler/11-26-2024/model.mlir",
    group="sdxl_scheduler",
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


ROCM_COMPILE_FLAGS = [
    "--iree-hal-target-backends=rocm",
    f"--iree-hip-target={rocm_chip}",
    "--iree-opt-const-eval=false",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-llvmgpu-enable-prefetch=true",
    "--iree-execution-model=async-external",
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics)",
    "--iree-scheduling-dump-statistics-format=json",
    "--iree-scheduling-dump-statistics-file=compilation_info.json",
]

###############################################################################
# CPU
###############################################################################


def test_compile_scheduler_cpu(sdxl_scheduler_mlir):
    VmfbManager.sdxl_scheduler_cpu_vmfb = iree_compile(
        sdxl_scheduler_mlir,
        CPU_COMPILE_FLAGS,
        Path(vmfb_dir)
        / Path("sdxl_scheduler_vmfbs")
        / Path(sdxl_scheduler_mlir.path.name).with_suffix(f".cpu.vmfb"),
    )


###############################################################################
# ROCM
###############################################################################


def test_compile_scheduler_rocm(sdxl_scheduler_mlir):
    VmfbManager.sdxl_scheduler_rocm_vmfb = iree_compile(
        sdxl_scheduler_mlir,
        ROCM_COMPILE_FLAGS,
        Path(vmfb_dir)
        / Path("sdxl_scheduler_vmfbs")
        / Path(sdxl_scheduler_mlir.path.name).with_suffix(f".rocm_{rocm_chip}.vmfb"),
    )
