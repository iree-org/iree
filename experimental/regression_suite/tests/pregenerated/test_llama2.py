# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers import *

###############################################################################
# Fixtures
###############################################################################

COMMON_FLAGS = [
    "--iree-input-type=none",
    "--iree-stream-resource-index-bits=64",
    "--iree-vm-target-index-bits=64",
    "--iree-stream-resource-max-allocation-size=3221225472",
]

llama2_7b_f16qi4_source = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/llama_regression/09152023/llama2_7b_int4_stripped.mlir",
    group="llama2_7b_f16qi4",
)


@pytest.fixture
def llama2_7b_f16qi4_rdna3_vulkan_vmfb(llama2_7b_f16qi4_source):
    return iree_compile(
        llama2_7b_f16qi4_source,
        "rdna3_vulkan",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=vulkan",
            "--iree-vulkan-target-triple=rdna3-unknown-linux",
        ],
    )


@pytest.fixture
def llama2_7b_f16qi4_host_cpu_vmfb(llama2_7b_f16qi4_source):
    return iree_compile(
        llama2_7b_f16qi4_source,
        "host_cpu",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu-features=host",
        ],
    )


###############################################################################
# Tests
###############################################################################


@pytest.mark.presubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_rdna3_vulkan
def test_step_rdna3_vulkan_stripped(llama2_7b_f16qi4_rdna3_vulkan_vmfb):
    iree_benchmark_module(
        llama2_7b_f16qi4_rdna3_vulkan_vmfb,
        device="vulkan",
        function="first_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ],
    )
    iree_benchmark_module(
        llama2_7b_f16qi4_rdna3_vulkan_vmfb,
        device="vulkan",
        function="second_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ]
        + (["--input=1x32x1x128xf16"] * 64),
    )


@pytest.mark.presubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_host_cpu
def test_step_host_cpu_stripped(llama2_7b_f16qi4_host_cpu_vmfb):
    iree_benchmark_module(
        llama2_7b_f16qi4_host_cpu_vmfb,
        device="local-task",
        function="first_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ],
    )
    iree_benchmark_module(
        llama2_7b_f16qi4_host_cpu_vmfb,
        device="local-task",
        function="second_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ]
        + (["--input=1x32x1x128xf16"] * 64),
    )
