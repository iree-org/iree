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
            "aarch64_vector_lowering.mlir",
            "apply_scale_lowering.mlir",
            "check_ir_before_llvm_conversion.mlir",
            "convert_to_llvm.mlir",
            "emit_vectorization_remarks.mlir",
            "hal_interface_bindings.mlir",
            "hal_interface_constants.mlir",
            "hal_interface_workgroup_info.mlir",
            "illegal_configuration.mlir",
            "linalg_transform.mlir",
            "materialize_aarch64_launch_configuration.mlir",
            "materialize_riscv_launch_configuration.mlir",
            "materialize_vmvx_launch_configuration.mlir",
            "materialize_x86_64_launch_configuration.mlir",
            "peel_and_vectorize.mlir",
            "pipeline_tests.mlir",
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
        # transform_dialect_*_spec is a an MLIR file that specifies a
        # transformation, it needs to be included as data.
        exclude = ["transform_dialect_codegen_bufferize_spec.mlir"],
    ),
    cfg = "//compiler:lit.cfg.py",
    data = ["transform_dialect_codegen_bufferize_spec.mlir"],
    tools = [
        "//tools:iree-compile",
        "//tools:iree-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
