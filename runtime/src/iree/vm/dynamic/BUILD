# Copyright 2023 The IREE Authors
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
    name = "api",
    hdrs = ["api.h"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/vm",
    ],
)

iree_runtime_cc_library(
    name = "module",
    srcs = ["module.c"],
    hdrs = ["module.h"],
    deps = [
        ":api",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal:dynamic_library",
        "//runtime/src/iree/vm",
    ],
)
