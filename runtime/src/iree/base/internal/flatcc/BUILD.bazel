# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library")
load("//build_tools/bazel:iree_flatcc.bzl", "iree_flatbuffer_c_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_runtime_cc_library(
    name = "building",
    hdrs = ["building.h"],
    deps = [
        ":dummy",
        ":parsing",
        "@com_github_dvidelabs_flatcc//:runtime",
    ],
)

iree_runtime_cc_library(
    name = "debugging",
    hdrs = ["debugging.h"],
    deps = [
        ":dummy",
        "@com_github_dvidelabs_flatcc//:runtime",
    ],
)

iree_runtime_cc_library(
    name = "parsing",
    hdrs = ["parsing.h"],
    deps = [
        ":dummy",
        "@com_github_dvidelabs_flatcc//:parsing",
    ],
)

iree_flatbuffer_c_library(
    name = "dummy",
    srcs = ["dummy.fbs"],
    flatcc_args = [
        "--reader",
        "--builder",
        "--verifier",
        "--json",
    ],
)
