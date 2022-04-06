# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for end-to-end IREE support of specific features to prevent regression.
# These should focus on support by IREE itself, not for issues with specific runner tools. Place
# those tests in iree/tools/test/

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

BACKEND_TESTS = [
    "dynamic_abs.mlir",
    "dynamic_add.mlir",
    "dynamic_dot.mlir",
    "dynamic_reduce_min.mlir",
    "dynamic_torch_index_select_high_rank.mlir",
    "dynamic_torch_index_select_negative.mlir",
    "dynamic_torch_index_select_scalar.mlir",
    "dynamic_torch_index_select_vector.mlir",
    "linalg_ops.mlir",
    "linalg_ext_ops.mlir",
    "strided_slice.mlir",
]

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "fill_i64.mlir",
            "globals.mlir",
            "libm_linking.mlir",
            "scalar.mlir",
            "trace_dispatch_tensors.mlir",
            "unused_args.mlir",
        ],
        include =
            ["*.mlir"],
        # TODO(#5897): enable these for codegen linalg on tensors/etc.
        exclude = [
            "dynamic_compare_and_select.mlir",
            "dynamic_dot_general.mlir",
            "dynamic_linalg_matmul_on_tensors.mlir",
            "dynamic_linalg_matmul_on_tensors_fuse_0.mlir",
            "dynamic_linalg_matmul_on_tensors_fuse_1.mlir",
            "dynamic_linalg_matmul_on_tensors_fuse_2.mlir",
            "linalg_quantized_matmul_vs_linalg_matmul.mlir",
            "lowering_config.mlir",
        ] + BACKEND_TESTS,
    ),
    tags = ["hostonly"],
    tools = [
        "//iree/tools:iree-opt",
        "//iree/tools:iree-run-mlir",
        "@llvm-project//lld",
        "@llvm-project//llvm:FileCheck",
    ],
)

iree_check_single_backend_test_suite(
    name = "check_regression_dylib-llvm-aot",
    srcs = [
        "lowering_config.mlir",
    ] + BACKEND_TESTS,
    compiler_flags = ["-iree-input-type=mhlo"],
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

iree_check_single_backend_test_suite(
    name = "check_regression_tosa_dylib-llvm-aot",
    srcs = [
        # -iree-input-type=tosa is interesting for this test because
        # it includes the -iree-linalg-quantized-matmul-to-matmul pass,
        # so this tests provides e2e coverage for that pass.
        "linalg_quantized_matmul_vs_linalg_matmul.mlir",
    ],
    compiler_flags = ["-iree-input-type=tosa"],
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

iree_check_single_backend_test_suite(
    name = "check_regression_vmvx",
    srcs = BACKEND_TESTS,
    compiler_flags = ["-iree-input-type=mhlo"],
    driver = "vmvx",
    target_backend = "vmvx",
)

iree_check_single_backend_test_suite(
    name = "check_regression_vulkan-spirv",
    srcs = BACKEND_TESTS,
    compiler_flags = ["-iree-input-type=mhlo"],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_regression_cuda",
    srcs = BACKEND_TESTS,
    compiler_flags = ["-iree-input-type=mhlo"],
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
