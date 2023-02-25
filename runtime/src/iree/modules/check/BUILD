# Copyright 2020 The IREE Authors
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
    name = "check",
    testonly = True,
    srcs = ["module.cc"],
    hdrs = ["module.h"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/modules/hal",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:cc",
    ],
)

iree_runtime_cc_test(
    name = "check_test",
    srcs = ["check_test.cc"],
    deps = [
        ":check",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal:span",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/modules/hal",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
        "//runtime/src/iree/tooling:device_util",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:cc",
        "//runtime/src/iree/vm/bytecode:module",
    ],
)
