# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "check_linalg_transform",
    srcs = ["linalg_transform.mlir"],
    cfg = "//tests:lit.cfg.py",
    # transform_dialect_xxx_spec are MLIR files that specify a transformation,
    # they need to be included as data.
    data = [
        "//tests/e2e/linalg_transform:transform_dialect_codegen_spec.mlir",
        "//tests/e2e/linalg_transform:transform_dialect_dispatch_spec.mlir",
    ],
    tags = [
        "hostonly",
    ],
    tools = [
        "//tools:iree-opt",
        "//tools:iree-run-mlir",
        "@llvm-project//lld",
        "@llvm-project//llvm:FileCheck",
    ],
)
