# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for end-to-end IREE support of entire models or their close derivatives.

load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = [
        "contraction-packing.mlir",
        "contraction-packing-and-dispatch.mlir",
        "eltwise_reduction_eltwise.mlir",
        "matmul.mlir",
    ],
    cfg = "//tests:lit.cfg.py",
    # transform dialect spec files are MLIR files that specify a transformation,
    # they need to be included as data.
    data = [
        "matmul_codegen_custom_dispatch_formation_spec.mlir",
        "matmul_codegen_default_spec.mlir",
    ],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
    ],
    tools = [
        "//tools:iree-benchmark-module",
        "//tools:iree-compile",
        "//tools:iree-opt",
        "//tools:iree-run-module",
        "@llvm-project//lld",
        "@llvm-project//llvm:FileCheck",
    ],
)
