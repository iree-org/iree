# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_build_test", "iree_runtime_cc_library")
load("//build_tools/bazel:iree_flatcc.bzl", "iree_flatbuffer_c_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

FLATCC_ARGS = [
    "--reader",
    "--builder",
    "--verifier",
    "--json",
]

iree_flatbuffer_c_library(
    name = "dispatch_def_c_fbs",
    srcs = ["dispatch_def.fbs"],
    flatcc_args = FLATCC_ARGS,
)

iree_build_test(
    name = "schema_build_test",
    targets = [
        ":dispatch_def_c_fbs",
    ],
)

iree_runtime_cc_library(
    name = "instruments",
    hdrs = [
        "dispatch.h",
    ],
)
