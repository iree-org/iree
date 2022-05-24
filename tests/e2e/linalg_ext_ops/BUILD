# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_check_single_backend_test_suite(
    name = "check_cuda",
    srcs = enforce_glob(
        # keep sorted
        [
            "reverse.mlir",
            "scan.mlir",
            "sort.mlir",
            "top-k.mlir",
        ],
        include = ["*.mlir"],
    ),
    driver = "cuda",
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backend = "cuda",
)

iree_check_single_backend_test_suite(
    name = "check_dylib-llvm-aot_dylib",
    srcs = enforce_glob(
        # keep sorted
        [
            "reverse.mlir",
            "scan.mlir",
            "sort.mlir",
            "top-k.mlir",
        ],
        include = ["*.mlir"],
    ),
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

iree_check_single_backend_test_suite(
    name = "check_vmvx_vmvx",
    srcs = enforce_glob(
        # keep sorted
        [
            "reverse.mlir",
            "scan.mlir",
            "sort.mlir",
            "top-k.mlir",
        ],
        include = ["*.mlir"],
    ),
    driver = "vmvx",
    target_backend = "vmvx",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = enforce_glob(
        # keep sorted
        [
            "reverse.mlir",
            # Top-k test disabled due to miscompile on vulkan.
            #    "top-k.mlir",
            "sort.mlir",
        ],
        include = ["*.mlir"],
        exclude = [
            # TODO(antiagainst): scan fails on Adreno GPUs due to driver bug.
            # Re-enable this once we have new devices with up-to-date drivers.
            "top-k.mlir",
            "scan.mlir",
        ],
    ),
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

test_suite(
    name = "check",
    tests = [
        ":check_cuda",
        ":check_dylib-llvm-aot_dylib",
        ":check_vmvx_vmvx",
        ":check_vulkan-spirv_vulkan",
    ],
)
