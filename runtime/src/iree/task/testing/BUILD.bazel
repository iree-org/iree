# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_runtime_cc_library(
    name = "task_test",
    testonly = True,
    hdrs = ["task_test.h"],
    deps = [
        "//runtime/src/iree/task",
        "//runtime/src/iree/testing:gtest",
    ],
)

iree_runtime_cc_library(
    name = "test_util",
    testonly = True,
    hdrs = ["test_util.h"],
    deps = [
        "//runtime/src/iree/task",
        "//runtime/src/iree/testing:gtest",
    ],
)
