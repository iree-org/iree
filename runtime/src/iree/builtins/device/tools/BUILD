# Copyright 2021 The IREE Authors
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
    name = "libdevice_benchmark",
    srcs = ["libdevice_benchmark.c"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/builtins/device",
        "//runtime/src/iree/testing:benchmark",
    ],
)

iree_runtime_cc_test(
    name = "libdevice_test",
    srcs = ["libdevice_test.cc"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/builtins/device",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)
