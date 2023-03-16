# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Smoke tests for the execution of tool binaries.

load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "compile_pipelines.mlir",
            "compile_to_continuation.mlir",
            "compile_to_phase.mlir",
            "executable_benchmarks.mlir",
            "executable_sources.mlir",
            "iree-benchmark-module.mlir",
            "iree-run-mlir.mlir",
            "iree-run-module.mlir",
            "iree-run-module-expected.mlir",
            "iree-run-module-outputs.mlir",
            "iree-run-trace.mlir",
            "multiple_args.mlir",
            "multiple_exported_functions.mlir",
            "null_values.mlir",
            "repeated_return.mlir",
            "scalars.mlir",
        ],
        include = ["*.mlir"],
    ),
    cfg = "//tools:lit.cfg.py",
    data = [
        "echo_npy.py",
        "iree-run-trace.yml",
    ],
    tags = [
        "driver=local-task",
        "driver=vulkan",
        "hostonly",
    ],
    tools = [
        "//tools:iree-benchmark-module",
        "//tools:iree-benchmark-trace",
        "//tools:iree-compile",
        "//tools:iree-opt",
        "//tools:iree-run-mlir",
        "//tools:iree-run-module",
        "//tools:iree-run-trace",
        "@llvm-project//lld",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

iree_lit_test_suite(
    name = "benchmark_flags",
    srcs = ["benchmark_flags.txt"],
    cfg = "//tools:lit.cfg.py",
    tags = [
        "hostonly",
    ],
    tools = [
        "//tools:iree-benchmark-module",
        "//tools:iree-compile",
        "@llvm-project//llvm:FileCheck",
    ],
)
