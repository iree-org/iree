# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for end-to-end IREE support specific to the vulkan-spirv lowering.
# TODO(ravishankarm): Reorganize these tests.

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = enforce_glob(
        [
            "compare.mlir",
            "conv.mlir",
            "dot_general.mlir",
            "log_plus_one.mlir",
            "pw_add_multiwg.mlir",
            "reduce.mlir",
            "vectorized_conv.mlir",
        ],
        include = ["*.mlir"],
        exclude = [
            "gemm.mlir",
        ],
    ),
    compiler_flags = ["-iree-input-type=mhlo"],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan_vectorized_conv",
    srcs = [
        "vectorized_conv.mlir",
    ],
    compiler_flags = [
        "-iree-input-type=mhlo",
        "-iree-vulkan-target-triple=valhall-unknown-android11",
    ],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)
