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
            "arithmetic_ops.mlir",
            "assignment_ops.mlir",
            "comparison_ops.mlir",
            "const_ops.mlir",
            "control_flow_ops.mlir",
            "conversion_ops.mlir",
            "func_attrs.mlir",
            "nesting.mlir",
            "structural_ops.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:iree-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
