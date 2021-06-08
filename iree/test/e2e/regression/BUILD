# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for end-to-end IREE support of specific features to prevent regression.
# These should focus on support by IREE itself, not for issues with specific runner tools. Place
# those tests in iree/tools/test/

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//iree:lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

BACKEND_TESTS = [
    "linalg_ops.mlir",
]

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "dynamic_abs.mlir",
            "dynamic_add.mlir",
            "dynamic_compare_and_select.mlir",
            "dynamic_dot.mlir",
            "dynamic_torch_index_select_high_rank.mlir",
            "dynamic_torch_index_select_negative.mlir",
            "dynamic_torch_index_select_scalar.mlir",
            "dynamic_torch_index_select_vector.mlir",
            "globals.mlir",
            "scalar.mlir",
            "trace_dispatch_tensors.mlir",
            "unused_args.mlir",
        ],
        include =
            ["*.mlir"],
        # TODO(#5897): enable these for codegen linalg on tensors/etc.
        exclude = [
            "dynamic_dot_general.mlir",
            "dynamic_linalg_matmul_on_tensors.mlir",
            "dynamic_linalg_matmul_on_tensors_fuse_0.mlir",
            "dynamic_linalg_matmul_on_tensors_fuse_1.mlir",
            "dynamic_linalg_matmul_on_tensors_fuse_2.mlir",
        ] + BACKEND_TESTS,
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-benchmark-module",
        "//iree/tools:iree-run-mlir",
        "//iree/tools:iree-translate",
    ],
    tags = ["hostonly"],
)

iree_check_single_backend_test_suite(
    name = "check_regression_linalg_ops_dylib-llvm-aot",
    srcs = BACKEND_TESTS,
    compiler_flags = ["-iree-input-type=mhlo"],
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

iree_check_single_backend_test_suite(
    name = "check_regression_linalg_ops_vmvx",
    srcs = BACKEND_TESTS,
    compiler_flags = ["-iree-input-type=mhlo"],
    driver = "vmvx",
    target_backend = "vmvx",
)

iree_check_single_backend_test_suite(
    name = "check_regression_linalg_ops_vulkan-spirv",
    srcs = BACKEND_TESTS,
    compiler_flags = ["-iree-input-type=mhlo"],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)
