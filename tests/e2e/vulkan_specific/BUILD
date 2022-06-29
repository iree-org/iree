# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for end-to-end IREE support specific to the vulkan-spirv lowering.

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
        "--iree-input-type=mhlo",
        "--iree-vulkan-target-triple=valhall-unknown-android11",
    ],
    driver = "vulkan",
    tags = [
        "manual",
        "notap",
        "vulkan_uses_vk_khr_shader_float16_int8",
    ],
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan_conv",
    srcs = [
        "conv.mlir",
    ],
    compiler_flags = [
        "--iree-input-type=mhlo",
        "--iree-vulkan-target-triple=valhall-unknown-android11",
    ],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)
