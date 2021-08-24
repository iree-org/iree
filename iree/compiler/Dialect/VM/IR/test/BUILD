# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
            "arithmetic_folding.mlir",
            "arithmetic_ops.mlir",
            "assignment_folding.mlir",
            "assignment_ops.mlir",
            "comparison_folding.mlir",
            "comparison_ops.mlir",
            "const_folding.mlir",
            "const_ops.mlir",
            "control_flow_folding.mlir",
            "control_flow_ops.mlir",
            "conversion_folding.mlir",
            "conversion_ops.mlir",
            "debug_folding.mlir",
            "debug_ops.mlir",
            "global_folding.mlir",
            "global_ops.mlir",
            "list_op_verification.mlir",
            "list_ops.mlir",
            "shift_ops.mlir",
            "structural_folding.mlir",
            "structural_ops.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-opt",
    ],
)
