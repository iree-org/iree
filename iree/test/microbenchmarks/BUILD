# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests compilation of microbenchmarks.

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

# Check host features compilation for Linalg microbenchmarks.
iree_check_single_backend_test_suite(
    name = "check_linalg_microbenchmarks_compilation",
    srcs = enforce_glob(
        # keep sorted
        [
            "linalg_mmt4d.mlir",
            "linalg_transpose.mlir",
        ],
        include = ["*.mlir"],
    ),
    compiler_flags = [
        "-iree-llvm-target-cpu-features=host",
    ],
    driver = "dylib",
    # Building and testing must be on the same architecture, which doesn't work
    # with remote execution in general.
    tags = [
        "hostonly",
        "local",
    ],
    target_backend = "dylib-llvm-aot",
)

# Check host features compilation for MHLO microbenchmarks.
iree_check_single_backend_test_suite(
    name = "check_mhlo_microbenchmarks_compilation",
    srcs = enforce_glob(
        # keep sorted
        [
            "mhlo_conv.mlir",
            "mhlo_dot_general.mlir",
            "mhlo_dot.mlir",
            "mhlo_fft_abs.mlir",
        ],
        include = ["*.mlir"],
    ),
    compiler_flags = [
        "-iree-input-type=mhlo",
        "-iree-llvm-target-cpu-features=host",
    ],
    driver = "dylib",
    # Building and testing must be on the same architecture, which doesn't work
    # with remote execution in general.
    tags = [
        "hostonly",
        "local",
    ],
    target_backend = "dylib-llvm-aot",
)

test_suite(
    name = "check_microbenchmarks",
    tests = [
        ":check_linalg_microbenchmarks_compilation",
        ":check_mhlo_microbenchmarks_compilation",
    ],
)
