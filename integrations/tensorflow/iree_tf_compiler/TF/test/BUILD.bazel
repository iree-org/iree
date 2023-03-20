# Copyright 2019 The IREE Authors
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
            "convert_to_mhlo.mlir",
            "direct_lowering.mlir",
            "lower_global_tensors.mlir",
            "lower_global_tensors_invalid.mlir",
            "propagate_resource_casts.mlir",
            "saved_model_to_iree_abi.mlir",
            "strip_asserts.mlir",
            "strip_metadata.mlir",
            "verify_fully_converted.mlir",
        ],
        include = ["*.mlir"],
    ),
    tools = [
        "//iree_tf_compiler:iree-tf-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
