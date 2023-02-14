# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT IREE_OUTPUT_FORMAT_C)
  return()
endif()
""",
)

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "arithmetic_ops_f32.mlir",
            "arithmetic_ops_i64.mlir",
            "arithmetic_ops.mlir",
            "assignment_ops_f32.mlir",
            "assignment_ops_i64.mlir",
            "assignment_ops.mlir",
            "buffer_ops.mlir",
            "buffer_ops_f32.mlir",
            "buffer_ops_f64.mlir",
            "buffer_ops_i64.mlir",
            "comparison_ops_f32.mlir",
            "comparison_ops_i64.mlir",
            "comparison_ops.mlir",
            "const_ops_f32.mlir",
            "const_ops_i64.mlir",
            "const_ops.mlir",
            "control_flow_ops.mlir",
            "conversion_ops_f32.mlir",
            "conversion_ops_i64.mlir",
            "conversion_ops.mlir",
            "func_op.mlir",
            "global_ops_f32.mlir",
            "global_ops_i64.mlir",
            "global_ops.mlir",
            "list_ops_i64.mlir",
            "list_ops.mlir",
            "shift_ops_i64.mlir",
            "shift_ops.mlir",
            "type_conversion.mlir",
        ],
        include = ["*.mlir"],
    ),
    cfg = "//compiler:lit.cfg.py",
    tools = [
        "//tools:iree-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
