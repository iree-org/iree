# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for end-to-end IREE support specific to the CUDA backend to be able to
# incrementally enable features.

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_check_single_backend_test_suite(
    name = "check_dylib-llvm-aot_dylib",
    srcs = [
        "dot_exp.mlir",
    ],
    compiler_flags = [
        "-iree-input-type=mhlo",
        "--iree-llvmcpu-use-tile-fuse-and-vectorize",
    ],
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)
