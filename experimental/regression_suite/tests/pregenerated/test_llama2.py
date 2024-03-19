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

llama2_7b_f16qi4_stripped_source = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/llama_regression/09152023/llama2_7b_int4_stripped.mlir",
    group="llama2_7b_f16qi4_stripped",
)

llama2_7b_f16qi4_source = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/llama_regression/llama2_7b_int4.mlir",
    group="llama2_7b_f16qi4",
)


@pytest.fixture
def llama2_7b_f16qi4_stripped_rdna3_vulkan_vmfb(llama2_7b_f16qi4_stripped_source):
    return iree_compile(
        llama2_7b_f16qi4_stripped_source,
        "rdna3_vulkan",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=vulkan-spirv",
            "--iree-vulkan-target-triple=rdna3-unknown-linux",
        ],
    )


@pytest.fixture
def llama2_7b_f16qi4_stripped_host_cpu_vmfb(llama2_7b_f16qi4_stripped_source):
    return iree_compile(
        llama2_7b_f16qi4_stripped_source,
        "host_cpu",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu-features=host",
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


@pytest.fixture
def llama2_7b_f16qi4_a100_vulkan_vmfb(llama2_7b_f16qi4_stripped_source):
    return iree_compile(
        llama2_7b_f16qi4_stripped_source,
        "a100_vulkan",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=vulkan-spirv",
            f"--iree-vulkan-target-triple=ampere-a100-linux",
        ],
    )


@pytest.fixture
def llama2_7b_f16qi4_stripped_sm80_cuda_vmfb(llama2_7b_f16qi4_stripped_source):
    return iree_compile(
        llama2_7b_f16qi4_stripped_source,
        "sm80_cuda",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=cuda",
            f"--iree-hal-cuda-llvm-target-arch=sm_80",
        ],
    )


@pytest.fixture
def llama2_7b_f16qi4_stripped_rdna3_rocm_vmfb(llama2_7b_f16qi4_stripped_source):
    return iree_compile(
        llama2_7b_f16qi4_stripped_source,
        "rdna3_rocm",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=rocm",
            "--iree-rocm-target-chip=gfx1100",
            "--iree-rocm-link-bc=true",
        ],
    )


@pytest.fixture
def llama2_7b_f16qi4_sm80_cuda_vmfb(llama2_7b_f16qi4_source):
    return iree_compile(
        llama2_7b_f16qi4_source,
        "sm70_cuda",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=cuda",
            f"--iree-hal-cuda-llvm-target-arch=sm_70",
        ],
    )


###############################################################################
# Performance
###############################################################################


@pytest.mark.presubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_rdna3_vulkan
def test_step_rdna3_vulkan_stripped(llama2_7b_f16qi4_stripped_rdna3_vulkan_vmfb):
    iree_benchmark_module(
        llama2_7b_f16qi4_stripped_rdna3_vulkan_vmfb,
        device="vulkan",
        function="first_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ],
    )
    iree_benchmark_module(
        llama2_7b_f16qi4_stripped_rdna3_vulkan_vmfb,
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
def test_step_host_cpu_stripped(llama2_7b_f16qi4_stripped_host_cpu_vmfb):
    iree_benchmark_module(
        llama2_7b_f16qi4_stripped_host_cpu_vmfb,
        device="local-task",
        function="first_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ],
    )
    iree_benchmark_module(
        llama2_7b_f16qi4_stripped_host_cpu_vmfb,
        device="local-task",
        function="second_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ]
        + (["--input=1x32x1x128xf16"] * 64),
    )


@pytest.mark.presubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_nvidia_a100
def test_step_sm80_cuda_stripped(llama2_7b_f16qi4_stripped_sm80_cuda_vmfb):
    iree_benchmark_module(
        llama2_7b_f16qi4_stripped_sm80_cuda_vmfb,
        device="cuda",
        function="first_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ],
    )
    iree_benchmark_module(
        llama2_7b_f16qi4_stripped_sm80_cuda_vmfb,
        device="cuda",
        function="second_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ]
        + (["--input=1x32x1x128xf16"] * 64),
    )


@pytest.mark.presubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_nvidia_a100
def test_step_a100_vulkan_stripped(llama2_7b_f16qi4_a100_vulkan_vmfb):
    iree_benchmark_module(
        llama2_7b_f16qi4_a100_vulkan_vmfb,
        device="vulkan",
        function="first_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ],
    )
    iree_benchmark_module(
        llama2_7b_f16qi4_a100_vulkan_vmfb,
        device="vulkan",
        function="second_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ]
        + (["--input=1x32x1x128xf16"] * 64),
    )


@pytest.mark.presubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_rdna3_rocm
def test_step_rdna3_rocm_stripped(llama2_7b_f16qi4_stripped_rdna3_rocm_vmfb):
    iree_benchmark_module(
        llama2_7b_f16qi4_stripped_rdna3_rocm_vmfb,
        device="rocm",
        function="first_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ],
    )
    iree_benchmark_module(
        llama2_7b_f16qi4_stripped_rdna3_rocm_vmfb,
        device="rocm",
        function="second_vicuna_forward",
        args=[
            "--input=1x1xi64",
        ]
        + (["--input=1x32x1x128xf16"] * 64),
    )


###############################################################################
# Correctness
###############################################################################


llama2_7b_f16qi4_first_input_cpu = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/llama_regression/llama2-7b-i4-golden-outputs/cpu/first_vicuna_forward_input.npy",
    group="llama2_7b_f16qi4_first_input_cpu",
)

llama2_7b_f16qi4_first_output_cpu = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/llama_regression/llama2-7b-i4-golden-outputs/cpu/first_vicuna_forward_output.npy",
    group="llama2_7b_f16qi4_first_output_cpu",
)

llama2_7b_f16qi4_second_input_cpu = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/llama_regression/llama2-7b-i4-golden-outputs/cpu/second_vicuna_forward_input.npy",
    group="llama2_7b_f16qi4_second_input_cpu",
)

llama2_7b_f16qi4_second_output_cpu = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/llama_regression/llama2-7b-i4-golden-outputs/cpu/second_vicuna_forward_output.npy",
    group="llama2_7b_f16qi4_second_output_cpu",
)


@pytest.mark.postsubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_host_cpu
def test_correctness_host_cpu(
    llama2_7b_f16qi4_host_cpu_vmfb,
    llama2_7b_f16qi4_first_input_cpu,
    llama2_7b_f16qi4_first_output_cpu,
    llama2_7b_f16qi4_second_input_cpu,
    llama2_7b_f16qi4_second_output_cpu,
):
    iree_run_module(
        llama2_7b_f16qi4_host_cpu_vmfb,
        device="local-task",
        function="first_vicuna_forward",
        args=[
            f"--input=@{llama2_7b_f16qi4_first_input_cpu.path}",
            f"--expected_output=@{llama2_7b_f16qi4_first_output_cpu.path}",
        ],
    )
    iree_run_module(
        llama2_7b_f16qi4_host_cpu_vmfb,
        device="local-task",
        function="second_vicuna_forward",
        args=[
            f"--input=@{llama2_7b_f16qi4_second_input_cpu.path}",
            f"--expected_output=@{llama2_7b_f16qi4_second_output_cpu.path}",
        ],
    )


llama2_7b_f16qi4_first_input_cuda = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/llama_regression/llama2-7b-i4-golden-outputs/cuda/first_vicuna_forward_input.npy",
    group="llama2_7b_f16qi4_first_input_cuda",
)

llama2_7b_f16qi4_first_output_cuda = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/llama_regression/llama2-7b-i4-golden-outputs/cuda/first_vicuna_forward_output.npy",
    group="llama2_7b_f16qi4_first_output_cuda",
)

llama2_7b_f16qi4_second_input_cuda = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/llama_regression/llama2-7b-i4-golden-outputs/cuda/second_vicuna_forward_input.npy",
    group="llama2_7b_f16qi4_second_input_cuda",
)

llama2_7b_f16qi4_second_output_cuda = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/llama_regression/llama2-7b-i4-golden-outputs/cuda/second_vicuna_forward_output.npy",
    group="llama2_7b_f16qi4_second_output_cuda",
)


@pytest.mark.postsubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_nvidia_a100
def test_correctness_sm80_cuda(
    llama2_7b_f16qi4_sm80_cuda_vmfb,
    llama2_7b_f16qi4_first_input_cuda,
    llama2_7b_f16qi4_first_output_cuda,
    llama2_7b_f16qi4_second_input_cuda,
    llama2_7b_f16qi4_second_output_cuda,
):
    iree_run_module(
        llama2_7b_f16qi4_sm80_cuda_vmfb,
        device="cuda",
        function="first_vicuna_forward",
        args=[
            f"--input=@{llama2_7b_f16qi4_first_input_cuda.path}",
            f"--expected_output=@{llama2_7b_f16qi4_first_output_cuda.path}",
        ],
    )
    iree_run_module(
        llama2_7b_f16qi4_sm80_cuda_vmfb,
        device="cuda",
        function="second_vicuna_forward",
        args=[
            f"--input=@{llama2_7b_f16qi4_second_input_cuda.path}",
            f"--expected_output=@{llama2_7b_f16qi4_second_output_cuda.path}",
        ],
    )
