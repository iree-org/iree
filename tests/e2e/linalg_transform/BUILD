# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "check_linalg_transform",
    srcs = ["linalg_transform.mlir"],
    cfg = "//tests:lit.cfg.py",
    # linalg_transform_spec is a an MLIR file that specifies a
    # transformation, it needs to be included as data.
    data = [
        "//tests/e2e/linalg_transform:linalg_transform_spec.mlir",
    ],
    tags = [
        "hostonly",
    ],
    tools = [
        "//tools:iree-run-mlir",
        "@llvm-project//lld",
        "@llvm-project//llvm:FileCheck",
    ],
)
