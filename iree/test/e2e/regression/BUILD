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
    "tensor_insert_slice.mlir",
]

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "globals.mlir",
            "scalar.mlir",
            "tensor_cast.mlir",
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
            "lowering_config.mlir",
            "matmul_f32.mlir",
            "matmul_i8.mlir",
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
    name = "check_regression_dylib-llvm-aot",
    srcs = [
        "lowering_config.mlir",
    ] + BACKEND_TESTS,
    compiler_flags = ["-iree-input-type=mhlo"],
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

# Test direct lowering of linalg.matmul (not going through linalg.mmt4d)
iree_check_single_backend_test_suite(
    name = "check_regression_matmul_default_dylib-llvm-aot",
    srcs = [
        "matmul_f32.mlir",
        "matmul_i8.mlir",
    ],
    driver = "dylib",
    opt_flags = [
        "--iree-flow-export-matmul-test-funcs",
    ],
    target_backend = "dylib-llvm-aot",
)

# Test linalg.matmul lowering via linalg.mmt4d.
#
# At the moment, --iree-flow-convert-linalg-matmul-to-mmt4d requires passing
# explicit M0, K0, N0 parameters. This is a temporary state. Eventually
# heuristics will be implemented to choose these values automatically, and the
# typical usage pattern will then be NOT to pass explicit values.
# For now, as we have to pass explicit values, that also pushes us to separate
# the test cases by data type (f32 vs i8) as they tend to want different M0,K0,N0
# values.
#
# Using the "wrong" M0,K0,N0 values does not prevent tests from succeeding
# but will typically prevent using the vectorization passes that we hope to be
# testing, so we make a best effort to pass plausible values so that vectorization
# code is exercised as much as possible.
#
# Note: --iree-flow-convert-linalg-matmul-to-mmt4d does not mean that all of the
# testcases are using mmt4d. This pass currently bails in many cases, such as when
# the matrix shapes are not known at compile time to be multiples of M0,K0,N0.
# At the moment (09/2021) only 2% (4 out of 238) of the testcases here are actually using mmt4d.
# The goal is to get to 100%.
#
# The choice of M0=8 K0=1 N0=8 reflects a typical f32 matmul kernel on aarch64
# (before the advent of the f32mm ISA extension). This is absolutely not future-proof and needs to be fixed
# by not having to specify explicit values here, as explained above.
iree_check_single_backend_test_suite(
    name = "check_regression_matmul_f32_mmt4d_8x1x8_dylib-llvm-aot",
    srcs = ["matmul_f32.mlir"],
    driver = "dylib",
    opt_flags = [
        "--iree-flow-export-matmul-test-funcs",
        "--iree-flow-convert-linalg-matmul-to-mmt4d=M0=8 K0=1 N0=8",
        "--iree-codegen-vectorize-linalg-mmt4d",
    ],
    target_backend = "dylib-llvm-aot",
)

# i8 matmul. Same comments as for the f32 matmul above. At the moment none of
# the testcases are actually using mmt4d because --iree-flow-convert-linalg-matmul-to-mmt4d
# currently only supports f32. The choice of M0=8 K0=1 N0=8 reflects a typical i8 matmul
# kernel on aarch64 with dot-product instructions (before the advent of the i8mm ISA extension).
# Same comments as above about all this being non-future proof and temporary.
iree_check_single_backend_test_suite(
    name = "check_regression_matmul_i8_mmt4d_8x4x8_dylib-llvm-aot",
    srcs = ["matmul_i8.mlir"],
    driver = "dylib",
    opt_flags = [
        "--iree-flow-export-matmul-test-funcs",
        "--iree-flow-convert-linalg-matmul-to-mmt4d=M0=8 K0=4 N0=8",
        "--iree-codegen-vectorize-linalg-mmt4d",
    ],
    target_backend = "dylib-llvm-aot",
)
