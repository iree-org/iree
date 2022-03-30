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
        [
            "hal_executable.mlir",
            "smoketest.mlir",
            "smoketest_linalg_transform.mlir",
            "streams.mlir",
        ],
        include = ["*.mlir"],
        # linalg_transform_spec is a an MLIR file that specifies a
        # transformation, it needs to be included as data.
        exclude = ["linalg_transform_spec.mlir"],
    ),
    data = ["linalg_transform_spec.mlir"],
    tools = [
        "//iree/tools:iree-check-module",
        "//iree/tools:iree-compile",
        "//iree/tools:iree-translate",
        "@llvm-project//llvm:FileCheck",
    ],
)
