# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_test")
load("//build_tools/bazel:cc_binary_benchmark.bzl", "cc_binary_benchmark")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary_benchmark(
    name = "mmt4d_benchmark",
    srcs = ["mmt4d_benchmark.c"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/builtins/mmt4d",
        "//runtime/src/iree/testing:benchmark",
    ],
)

iree_runtime_cc_test(
    name = "mmt4d_test",
    srcs = ["mmt4d_test.cc"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/builtins/mmt4d",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)
