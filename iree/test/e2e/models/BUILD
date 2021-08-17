# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for end-to-end IREE support of entire models or their close derivatives.

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//iree:lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

CHECK_FRAMEWORK_TESTS = [
    "bert_encoder_unrolled_fake_weights.mlir",
    "mobilenetv3_fake_weights.mlir",
]

iree_lit_test_suite(
    name = "lit",
    size = "medium",
    srcs = enforce_glob(
        [
            "collatz.mlir",
            "edge_detection.mlir",
            "fragment_000.mlir",
            "fullyconnected.mlir",
            "mnist_fake_weights.mlir",
            "resnet50_fake_weights.mlir",
            "unidirectional_lstm.mlir",
        ],
        include =
            ["*.mlir"],
        exclude = CHECK_FRAMEWORK_TESTS,
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-run-mlir",
    ],
    tags = [
        "hostonly",
        "optonly",  # swiftshader is too slow in dbg
    ],
)

iree_check_single_backend_test_suite(
    name = "check_linalg_on_tensors_dylib-llvm-aot_dylib",
    srcs = CHECK_FRAMEWORK_TESTS,
    compiler_flags = ["-iree-input-type=mhlo"],
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

# TODO(#5880): VMVX is not yet fast enough to run the larger model tests.
# iree_check_single_backend_test_suite(
#     name = "check_vmvx_vmvx",
#     srcs = CHECK_FRAMEWORK_TESTS,
#     driver = "vmvx",
#     target_backend = "vmvx",
# )

iree_check_single_backend_test_suite(
    name = "check_linalg_on_tensors_vulkan-spirv_vulkan",
    timeout = "long",
    srcs = CHECK_FRAMEWORK_TESTS,
    compiler_flags = ["-iree-input-type=mhlo"],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_linalg_on_tensors_cuda_cuda",
    timeout = "long",
    srcs = CHECK_FRAMEWORK_TESTS,
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
