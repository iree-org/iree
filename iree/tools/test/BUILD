# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Smoke tests for the execution of tool binaries.

load("//iree:lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "iree-benchmark-module.mlir",
            "iree-run-mlir.mlir",
            "iree-run-module.mlir",
            "iree-run-trace.mlir",
            "multiple_args.mlir",
            "multiple_exported_functions.mlir",
            "repeated_return.mlir",
            "scalars.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        ":iree-run-trace.yaml",
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-benchmark-module",
        "//iree/tools:iree-run-mlir",
        "//iree/tools:iree-run-module",
        "//iree/tools:iree-run-trace",
        "//iree/tools:iree-translate",
    ],
    tags = [
        "hostonly",
        "notap",  # Disabled due to b/203702082
    ],
)

iree_lit_test_suite(
    name = "benchmark_flags",
    srcs = ["benchmark_flags.txt"],
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-benchmark-module",
        "//iree/tools:iree-translate",
    ],
    tags = [
        "hostonly",
    ],
)
