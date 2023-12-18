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

argmax_ukernel_source = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/ukernel_regression/20231217/argmax/argmax_3d_linalg.mlir",
    group="argmax_ukernel",
)


@pytest.fixture
def argmax_ukernel_rdna3_rocm_vmfb(argmax_ukernel_source):
    return iree_compile(
        argmax_ukernel_source,
        "rdna3_rocm",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=rocm",
            "--iree-rocm-target-chip=gfx1100",
            "--iree-rocm-link-bc=true",
            "--iree-rocm-enable-ukernels=all",
        ],
    )


###############################################################################
# Correctness
###############################################################################

# Generation script:
# argmax_input_f16 = np.random.normal(size=[2, 4, 33000]).astype(np.float32)
# argmax_output_f16 = np.argmax(argmax_input_f16,axis=-1).astype(np.float32)
# argmax_input_f32 = np.random.normal(size=[2, 4, 33000]).astype(np.float32)
# argmax_output_f32 = np.argmax(argmax_input_f32,axis=-1).astype(np.float32)
# TODO: Currently forcing sitofp (i32 -> f32) and (i64 -> f32) because expected_output
#       cannot compare signless i64 from vmfb and by default si64 from npy.

argmax_input_f16 = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/ukernel_regression/20231217/argmax/argmax_input_f16.npy",
    group="argmax_ukernel",
)

argmax_output_f16 = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/ukernel_regression/20231217/argmax/argmax_output_f16.npy",
    group="argmax_ukernel",
)

argmax_input_f32 = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/ukernel_regression/20231217/argmax/argmax_input_f32.npy",
    group="argmax_ukernel",
)

argmax_output_f32 = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/ukernel_regression/20231217/argmax/argmax_output_f32.npy",
    group="argmax_ukernel",
)


@pytest.mark.presubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_rdna3_rocm
def test_correctness_rnda3_rocm(
    argmax_ukernel_rdna3_rocm_vmfb,
    argmax_input_f16,
    argmax_output_f16,
    argmax_input_f32,
    argmax_output_f32,
):
    iree_run_module(
        argmax_ukernel_rdna3_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f16i32",
        args=[
            f"--input=@{argmax_input_f16.path}",
            f"--expected_output=@{argmax_output_f16.path}",
        ],
    )
    iree_run_module(
        argmax_ukernel_rdna3_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f16i64",
        args=[
            f"--input=@{argmax_input_f16.path}",
            f"--expected_output=@{argmax_output_f16.path}",
        ],
    )

    iree_run_module(
        argmax_ukernel_rdna3_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f32i32",
        args=[
            f"--input=@{argmax_input_f32.path}",
            f"--expected_output=@{argmax_output_f32.path}",
        ],
    )
    iree_run_module(
        argmax_ukernel_rdna3_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f32i64",
        args=[
            f"--input=@{argmax_input_f32.path}",
            f"--expected_output=@{argmax_output_f32.path}",
        ],
    )
