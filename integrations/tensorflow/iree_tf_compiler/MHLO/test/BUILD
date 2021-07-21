# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@iree//iree:lit_test.bzl", "iree_lit_test_suite")
load("@iree//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "emit_default_iree_abi.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree_tf_compiler:iree-tf-opt",
        "@iree//iree/tools:IreeFileCheck",
    ],
    driver = "@iree//iree/tools:run_lit.sh",
)
