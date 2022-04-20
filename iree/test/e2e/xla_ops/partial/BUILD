# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests of end-to-end IREE support for support of specific subsets of ops in the
# XLA HLO dialect. This is for cases where some variation (e.g. tensor element
# types) of an op is not supported on all backends. When the test is supported
# on all backends it should be moved into the corresponding op test in the main
# xla_ops direcotry. Each test file should have a name matching the
# corresponding XLA HLO op with a suffix indicating the subset it tests. Only
# the functionality of that op should be tessted (though it may make use of
# other ops where necessary). Tests should be written using the IREE Check
# framework and should always pass on the reference VMVX backend. See
# https://github.com/google/iree/blob/main/docs/developers/developing_iree/testing_guide.md#iree-core-end-to-end-tests.

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan_f16",
    srcs = [
        "add_f16.mlir",
        "dot_f16.mlir",
    ],
    compiler_flags = [
        "-iree-input-type=mhlo",
        "-iree-vulkan-target-triple=valhall-unknown-android11",
    ],
    driver = "vulkan",
    tags = [
        "manual",
        "notap",
        "vulkan_uses_vk_khr_shader_float16_int8",
    ],
    target_backend = "vulkan-spirv",
)
