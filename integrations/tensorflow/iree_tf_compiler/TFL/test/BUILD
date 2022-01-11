# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:lit_test.bzl", "iree_lit_test_suite")
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
            "convert_metadata.mlir",
            "flex_ops.mlir",
            "lower_global_tensors.mlir",
            "retain_call_once_funcs.mlir",
            "strip_metadata.mlir",
            "verify_fully_converted.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//build_tools/bazel:IreeFileCheck",
        "//iree_tf_compiler:iree-opt-tflite",
    ],
    driver = "//build_tools/bazel:run_lit.sh",
)
