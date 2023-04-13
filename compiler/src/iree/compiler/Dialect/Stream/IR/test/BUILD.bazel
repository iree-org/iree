# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "async_folding.mlir",
            "async_ops.mlir",
            "builtin_ops.mlir",
            "channel_ops.mlir",
            "cmd_folding.mlir",
            "cmd_ops.mlir",
            "executable_ops.mlir",
            "resource_folding.mlir",
            "resource_ops.mlir",
            "tensor_folding.mlir",
            "tensor_ops.mlir",
            "timepoint_folding.mlir",
            "timepoint_ops.mlir",
        ],
        include = ["*.mlir"],
    ),
    cfg = "//compiler:lit.cfg.py",
    tools = [
        "//tools:iree-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
