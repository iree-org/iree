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
            "check_ir_before_llvm_conversion.mlir",
            "hal_interface_bindings.mlir",
            "hal_interface_constants.mlir",
            "hal_interface_workgroup_info.mlir",
            "illegal_configuration.mlir",
            "materialize_double_tiling_expert_configuration.mlir",
            "materialize_launch_configuration.mlir",
            "synchronize_symbol_visibility.mlir",
            "test_config_mmt4d.mlir",
            "tile_and_vectorize.mlir",
            "tile_fuse_and_vectorize.mlir",
            "unfused_fma.mlir",
            "vector_contract_to_aarch64_asm.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:iree-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
