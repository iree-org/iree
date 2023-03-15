# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library", "iree_runtime_cc_test")
load("//build_tools/bazel:cc_binary_benchmark.bzl", "cc_binary_benchmark")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_runtime_cc_library(
    name = "ukernel_test_utils",
    srcs = ["ukernel_test_utils.cc"],
    hdrs = ["ukernel_test_utils.h"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/builtins/ukernel:headers",
        "//runtime/src/iree/schemas:cpu_data",
    ],
)

cc_binary_benchmark(
    name = "mmt4d_benchmark",
    srcs = ["mmt4d_benchmark.c"],
    deps = [
        ":ukernel_test_utils",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/builtins/ukernel",
        "//runtime/src/iree/testing:benchmark",
    ],
)

iree_runtime_cc_test(
    name = "mmt4d_test",
    srcs = ["mmt4d_test.cc"],
    deps = [
        ":ukernel_test_utils",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/builtins/ukernel",
        "//runtime/src/iree/testing:gtest",
    ],
)

cc_binary_benchmark(
    name = "pack_benchmark",
    srcs = ["pack_benchmark.c"],
    deps = [
        ":ukernel_test_utils",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/builtins/ukernel",
        "//runtime/src/iree/testing:benchmark",
    ],
)

iree_runtime_cc_test(
    name = "pack_test",
    srcs = ["pack_test.cc"],
    deps = [
        ":ukernel_test_utils",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/builtins/ukernel",
        "//runtime/src/iree/testing:gtest",
    ],
)

cc_binary_benchmark(
    name = "unpack_benchmark",
    srcs = ["unpack_benchmark.c"],
    deps = [
        ":ukernel_test_utils",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/builtins/ukernel",
        "//runtime/src/iree/testing:benchmark",
    ],
)

iree_runtime_cc_test(
    name = "unpack_test",
    srcs = ["unpack_test.cc"],
    deps = [
        ":ukernel_test_utils",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/builtins/ukernel",
        "//runtime/src/iree/testing:gtest",
    ],
)
