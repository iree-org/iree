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
llama_real_weights = fetch_source_fixture(
    "https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/8b_f16.irpa",
    group="llama_fp16_8b",
)

llama_mlir = fetch_source_fixture(
    "https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/llm-dev/llama3_8b/8b_f16_decomposed_11_22.mlir",
    group="llama_fp16_8b",
)


ROCM_COMPILE_FLAGS = [
    "--iree-hal-target-backends=rocm",
    f"--iree-hip-target={rocm_chip}",
    "--iree-dispatch-creation-enable-aggressive-fusion=true",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-opt-aggressively-propagate-transposes=true",
    "--iree-opt-data-tiling=false",
    "--iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))'",
    "--iree-stream-resource-memory-model=discrete",
    "--iree-hip-legacy-sync=false",
    "--iree-hal-indirect-command-buffers=true",
    "--iree-hal-memoization=true",
    "--iree-opt-strip-assertions",
]

###############################################################################
# ROCM
###############################################################################


def test_compile_llama_rocm(llama_mlir):
    VmfbManager.sdxl_clip_rocm_vmfb = iree_compile(
        llama_mlir,
        ROCM_COMPILE_FLAGS,
        Path(vmfb_dir)
        / Path("llama_vmfbs")
        / Path(llama_mlir.path.name).with_suffix(f".rocm_{rocm_chip}.vmfb"),
    )


# TODO: Add run support
# @pytest.mark.depends(on=["test_compile_llama_rocm"])
# def test_run_clip_rocm(LLAMA_COMMON_RUN_FLAGS, llama_real_weights):
#    return iree_run_module(
#        VmfbManager.llama_rocm_vmfb,
#        device="hip",
#        function="prefill",
#        args=[
#            f"--parameters=model={llama_real_weights.path}",
#        ]
#        + LLAMA_COMMON_RUN_FLAGS,
#    )
