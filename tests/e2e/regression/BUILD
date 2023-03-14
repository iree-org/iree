# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for end-to-end IREE support of specific features to prevent regression.
# These should focus on support by IREE itself, not for issues with specific runner tools.
# Place those tests in tools/test/

load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
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
    "i1_inlined_constant.mlir",
    "linalg_ops.mlir",
    "reduction_broadcast_elementwise.mlir",
    "softmax.mlir",
    "strided_slice.mlir",
    "transpose.mlir",
]

iree_lit_test_suite(
    name = "lit",
    srcs = [
        "fill_i64.mlir",
        "globals.mlir",
        "libm_linking.mlir",
        "scalar.mlir",
        "trace_dispatch_tensors.mlir",
        "unused_args.mlir",
    ],
    cfg = "//tests:lit.cfg.py",
    tags = [
        "driver=local-task",
        "driver=vulkan",
        "hostonly",
    ],
    tools = [
        "//tools:iree-opt",
        "//tools:iree-run-mlir",
        "@llvm-project//lld",
        "@llvm-project//llvm:FileCheck",
    ],
)

iree_check_single_backend_test_suite(
    name = "check_regression_llvm-cpu",
    srcs = [
        "associative_reordering.mlir",
        "layernorm.mlir",
        "lowering_config.mlir",
        "pack_pad_transpose_1x9_into_2x4x8x4_issue_12546.mlir",
    ] + BACKEND_TESTS,
    compiler_flags = ["--iree-input-type=mhlo"],
    driver = "local-task",
    target_backend = "llvm-cpu",
)

iree_check_single_backend_test_suite(
    name = "check_regression_tosa_llvm-cpu",
    srcs = [
        # --iree-input-type=tosa is interesting for this test because
        # it includes the --iree-linalg-quantized-matmul-to-matmul pass,
        # so this tests provides e2e coverage for that pass.
        "linalg_quantized_matmul_vs_linalg_matmul.mlir",
    ],
    compiler_flags = ["--iree-input-type=tosa"],
    driver = "local-task",
    target_backend = "llvm-cpu",
)

iree_check_single_backend_test_suite(
    name = "check_regression_vmvx",
    srcs = [
        "layernorm.mlir",
    ] + BACKEND_TESTS,
    compiler_flags = ["--iree-input-type=mhlo"],
    driver = "local-task",
    target_backend = "vmvx",
)

iree_check_single_backend_test_suite(
    name = "check_regression_vulkan-spirv",
    # TODO(#10024): Fix layernorm.mlir on Pixel 6 and put in BACKEND_TESTS.
    srcs = BACKEND_TESTS,
    compiler_flags = ["--iree-input-type=mhlo"],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_regression_cuda",
    srcs = [
        "large_reduction.mlir",
        "layernorm.mlir",
    ] + BACKEND_TESTS,
    compiler_flags = ["--iree-input-type=mhlo"],
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
    name = "aggressive_fusion_test_cuda",
    srcs = [
        "softmax.mlir",
        "softmax_large.mlir",
    ],
    compiler_flags = [
        "--iree-flow-enable-aggressive-fusion",
    ],
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
    name = "disable_demote_f64_to_f32",
    srcs = [
        "disable_demote_f64_to_f32.mlir",
    ],
    compiler_flags = ["-iree-flow-demote-f64-to-f32=false"],
    driver = "local-task",
    target_backend = "llvm-cpu",
)

iree_check_single_backend_test_suite(
    name = "aggressive_fusion_test",
    srcs = [
        "softmax.mlir",
    ],
    compiler_flags = [
        "--iree-flow-enable-aggressive-fusion",
    ],
    driver = "local-task",
    target_backend = "llvm-cpu",
)

# Compilation-only (no driver specified, the default None means compile only)
# regression testcases for VMVX backend with microkernels. Indeed we have had
# a couple of hard-to-diagnose issues that were purely compilation issue, no
# particular concern about correctness once compilation succeeds, and having
# compilation-only tests allows us to import the original unmodified testcase.
iree_check_single_backend_test_suite(
    name = "check_regression_compilation_only_vmvx_ukernels",
    srcs = [
        "dynamic_matmuls_on_same_accumulator_issue_12060.mlir",
        "dynamic_tosa_clamp.mlir",
        "dynamic_tosa_quantized_fully_connected_issue_10859.mlir",
    ],
    compiler_flags = [
        "--iree-input-type=tosa",
        "--iree-vmvx-enable-microkernels",
    ],
    target_backend = "vmvx",
)
