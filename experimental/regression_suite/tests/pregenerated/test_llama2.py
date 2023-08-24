# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers import *

llama2_7b_f16qi4_source = fetch_source_fixture("foobar")


@pytest.fixture
def llama2_7b_f16qi4_rdna3_vmfb(llama2_7b_f16qi4_source):
    return compile_iree(
        llama2_7b_f16qi4_source,
        flags=[
            "--iree-input-type=none",
            "--iree-hal-target-backends=vulkan",
            "--iree-llvmcpu-target-cpu-features=host",
            "--iree-stream-resource-index-bits=64",
            "--iree-vm-target-index-bits=64",
            "--iree-vulkan-target-triple=rdna3-unknown-linux",
            "--iree-opt-const-expr-hoisting=false",
            "--iree-stream-resource-max-allocation-size=3221225472",
        ],
    )


@pytest.mark.presubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_rdna3_vulkan
def test_step_rdna3_vulkan(llama2_7b_f16qi4_rdna3_vmfb):
    golden_inputs = "TODO"
    expected_outputs = "TODO"
    execute_module(
        llama2_7b_f16qi4_rdna3_vmfb,
        driver="vulkan",
        inputs=golden_inputs,
        expected_outputs=expected_outputs,
    )
    benchmark_module(
        llama2_7b_f16qi4_rdna3_vmfb,
        driver="vulkan",
        inputs=golden_inputs,
    )
