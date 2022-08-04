# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests of end-to-end IREE support for some high level ops.
# See https://github.com/iree-org/iree/blob/main/docs/developers/developing_iree/testing_guide.md#iree-core-end-to-end-tests.

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

SRCS = enforce_glob(
    [
        "layernorm.mlir",
        "reduction_broadcast_elementwise.mlir",
        "softmax.mlir",
    ],
    include = ["*.mlir"],
)

iree_check_single_backend_test_suite(
    name = "check_dylib-llvm-aot_dylib",
    srcs = SRCS,
    compiler_flags = ["--iree-input-type=tosa"],
    driver = "local-task",
    target_backend = "llvm-cpu",
)

iree_check_single_backend_test_suite(
    name = "check_vmvx_vmvx",
    srcs = SRCS,
    compiler_flags = [
        "--iree-input-type=tosa",
    ],
    driver = "local-task",
    target_backend = "vmvx",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = SRCS,
    compiler_flags = ["--iree-input-type=tosa"],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

test_suite(
    name = "check",
    tests = [
        ":check_dylib-llvm-aot_dylib",
        ":check_vmvx_vmvx",
        ":check_vulkan-spirv_vulkan",
    ],
)
