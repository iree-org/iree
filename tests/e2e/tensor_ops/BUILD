# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = [
        "tensor_cast.mlir",
    ],
    cfg = "//tests:lit.cfg.py",
    tags = [
        "driver=local-task",
        "driver=vulkan",
        "hostonly",
    ],
    tools = [
        "//tools:iree-benchmark-module",
        "//tools:iree-compile",
        "//tools:iree-run-mlir",
        "@llvm-project//llvm:FileCheck",
    ],
)

iree_check_single_backend_test_suite(
    name = "check_llvm-cpu_local-task",
    srcs = enforce_glob(
        # keep sorted
        [
            "extract_slice.mlir",
            "pack.mlir",
            "tensor_insert_slice.mlir",
        ],
        include = ["*.mlir"],
        exclude = [
            "tensor_cast.mlir",
        ],
    ),
    driver = "local-task",
    target_backend = "llvm-cpu",
)

iree_check_single_backend_test_suite(
    name = "check_vmvx_local-task",
    srcs = enforce_glob(
        # keep sorted
        [
            "extract_slice.mlir",
            "tensor_insert_slice.mlir",
        ],
        include = ["*.mlir"],
        exclude = [
            "pack.mlir",
            "tensor_cast.mlir",
        ],
    ),
    driver = "local-task",
    target_backend = "vmvx",
)

iree_check_single_backend_test_suite(
    name = "check_cuda",
    srcs = enforce_glob(
        # keep sorted
        [
            "extract_slice.mlir",
            "tensor_insert_slice.mlir",
        ],
        include = ["*.mlir"],
        exclude = [
            "pack.mlir",
            "tensor_cast.mlir",
        ],
    ),
    driver = "cuda",
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backend = "cuda",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = enforce_glob(
        # keep sorted
        [
            "extract_slice.mlir",
            "tensor_insert_slice.mlir",
        ],
        include = ["*.mlir"],
        exclude = [
            "pack.mlir",
            "tensor_cast.mlir",
        ],
    ),
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)
