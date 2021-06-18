# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Testing utilities for IREE.

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "benchmark",
    testonly = True,
    srcs = [
        "benchmark_full.cc",
    ],
    hdrs = [
        "benchmark.h",
    ],
    deps = [
        "//iree/base",
        "//iree/base:tracing",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "benchmark_main",
    testonly = True,
    srcs = ["benchmark_main.c"],
    deps = [
        ":benchmark",
        "//iree/base/internal:flags",
    ],
)

iree_cmake_extra_content(
    content = """
# TODO(#3848): gtest library has implicit thread dependency from absl.
if(NOT ${IREE_ENABLE_THREADING})
  return()
endif()
""",
    inline = True,
)

cc_library(
    name = "gtest",
    testonly = True,
    hdrs = [
        "gtest.h",
        "status_matchers.h",
    ],
    deps = [
        "//iree/base:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gtest_main",
    testonly = True,
    srcs = ["gtest_main.cc"],
    tags = ["keep_dep"],
    deps = [
        ":gtest",
        "//iree/base/internal:flags",
        "@com_google_googletest//:gtest",
    ],
)
