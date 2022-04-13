# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for common transforms.

load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        # keep sorted
        [
            "check_ir_before_llvm_conversion.mlir",
            "hal_interface_bindings.mlir",
            "hal_interface_constants.mlir",
            "hal_interface_workgroup_info.mlir",
            "illegal_configuration.mlir",
            "linalg_transform.mlir",
            "materialize_aarch64_launch_configuration.mlir",
            "materialize_riscv_launch_configuration.mlir",
            "materialize_x86_64_launch_configuration.mlir",
            "synchronize_symbol_visibility.mlir",
            "test_config_mmt4d.mlir",
            "tile_fuse_and_vectorize.mlir",
            "transpose_avx2_lowering.mlir",
            "unfused_fma.mlir",
            "vector_contract_to_arm_asm.mlir",
            "vector_contract_to_arm_intrinsics.mlir",
            "verify_linalg_transform_legality.mlir",
        ],
        include = ["*.mlir"],
        # linalg_transform_spec is a an MLIR file that specifies a
        # transformation, it needs to be included as data.
        exclude = ["linalg_transform_spec.mlir"],
    ),
    data = ["linalg_transform_spec.mlir"],
    tools = [
        "//iree/tools:iree-compile",
        "//iree/tools:iree-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
