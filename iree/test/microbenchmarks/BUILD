# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests compilation of microbenchmarks.

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    size = "medium",
    srcs = enforce_glob(
        [
            "linalg_transpose.mlir",
        ],
        include = ["*.mlir"],
        exclude =
            [
                "linalg_mmt4d.mlir",
                "mhlo_conv.mlir",
                "mhlo_dot_general.mlir",
                "mhlo_dot.mlir",
                "mhlo_fft_abs.mlir",
            ],
    ),
    tags = [
        "hostonly",
    ],
    tools = [
        "//iree/tools:iree-run-mlir",
    ],
)
