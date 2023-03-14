# Copyright 2022 The IREE Authors
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
    name = "inline",
    srcs = [
        "module.c",
    ],
    hdrs = [
        "module.h",
    ],
    textual_hdrs = [
        "exports.inl",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/modules/hal:types",
        "//runtime/src/iree/modules/hal/utils:buffer_diagnostics",
        "//runtime/src/iree/vm",
    ],
)
