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
load("//build_tools/bazel:iree_trace_runner_test.bzl", "iree_generated_trace_runner_test")

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
        "linalg_quantized_matmul_vs_linalg_matmul.mlir",
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

py_binary(
    name = "generate_e2e_matmul_tests",
    srcs = ["generate_e2e_matmul_tests.py"],
)

[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_%s_small" % lhs_rhs_type,
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=small",
    ],
    target_backends_and_drivers = [
        ("dylib-llvm-aot", "dylib"),
        ("vmvx", "vmvx"),
    ],
    trace_runner = "//iree/tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
]]

# Test asm
[iree_generated_trace_runner_test(
    name = "e2e_matmul_mmt4d_%s_small" % lhs_rhs_type,
    compiler_flags = [
        "--iree-flow-mmt4d-target-options=enable_generic_slow #pass_options_variant#",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=small",
    ],
    target_backends_and_drivers = [
        ("dylib-llvm-aot", "dylib"),
    ],
    target_cpu_features_variants = ["default"] +
                                   (["aarch64:+dotprod"] if lhs_rhs_type == "i8" else []),
    trace_runner = "//iree/tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
]]

[iree_generated_trace_runner_test(
    name = "e2e_matmul_mmt4d_%s_large" % lhs_rhs_type,
    compiler_flags = [
        "--iree-flow-mmt4d-target-options=enable_generic_slow #pass_options_variant#",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=large",
    ],
    target_backends_and_drivers = [
        ("dylib-llvm-aot", "dylib"),
    ],
    target_cpu_features_variants = ["default"] +
                                   (["aarch64:+dotprod"] if lhs_rhs_type == "i8" else []),
    trace_runner = "//iree/tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
]]

# Test intrinsics. No need to run vmvx again, since it isn't affected by this
# codegen flag. No need to run "large" sizes, since this only differs from other
# tests in ways that are orthogonal to problem sizes.
[iree_generated_trace_runner_test(
    name = "e2e_matmul_mmt4d_%s_intrinsics_%s" % (lhs_rhs_type, size),
    compiler_flags = [
        "--iree-codegen-mmt4d-use-intrinsics",
        "--iree-flow-mmt4d-target-options=enable_generic_slow #pass_options_variant#",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=%s" % size,
    ],
    target_backends_and_drivers = [
        ("dylib-llvm-aot", "dylib"),
    ],
    target_cpu_features_variants = ["default"] +
                                   (["aarch64:+dotprod"] if lhs_rhs_type == "i8" else []),
    trace_runner = "//iree/tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
] for size in [
    "small",
]]
