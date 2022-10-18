# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library", "iree_runtime_cc_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_runtime_cc_library(
    name = "arena",
    srcs = ["arena.cc"],
    hdrs = ["arena.h"],
    deps = [
        "//runtime/src/iree/base",
    ],
)

iree_runtime_cc_test(
    name = "arena_test",
    srcs = ["arena_test.cc"],
    deps = [
        ":arena",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_library(
    name = "intrusive_list",
    hdrs = [
        "intrusive_list.h",
        "intrusive_list_unique_ptr.inc",
    ],
    deps = [
        "//runtime/src/iree/base",
    ],
)

iree_runtime_cc_test(
    name = "intrusive_list_test",
    srcs = [
        "intrusive_list_test.cc",
        "intrusive_list_unique_ptr_test.cc",
    ],
    deps = [
        ":intrusive_list",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_library(
    name = "ref_ptr",
    hdrs = ["ref_ptr.h"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal",
    ],
)

iree_runtime_cc_test(
    name = "ref_ptr_test",
    timeout = "short",
    srcs = ["ref_ptr_test.cc"],
    deps = [
        ":ref_ptr",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)
