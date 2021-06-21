# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for common transforms.

load("//iree:lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "hal_interface_bindings.mlir",
            "hal_interface_constants.mlir",
            "hal_interface_workgroup_info.mlir",
            "linalg_vectorize.mlir",
            "materialize_launch_configuration.mlir",
            "matmul_vectorization.mlir",
            "pad_linalg_workgroup_tiles.mlir",
            "plan_conv_loop_order.mlir",
            "tile_pad_and_vectorize_workgroups.mlir",
            "unfused_fma.mlir",
            "vector_contract_to_aarch64_asm.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-opt",
    ],
)
