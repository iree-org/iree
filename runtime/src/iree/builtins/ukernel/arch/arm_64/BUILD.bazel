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
    name = "mmt4d_arm_64",
    hdrs = [
        "mmt4d_arm_64.h",
    ],
)

iree_runtime_cc_library(
    name = "pack_arm_64",
    hdrs = [
        "pack_arm_64.h",
    ],
)

iree_runtime_cc_library(
    name = "query_tile_sizes_arm_64",
    hdrs = [
        "query_tile_sizes_arm_64.h",
    ],
)

iree_runtime_cc_library(
    name = "unpack_arm_64",
    hdrs = [
        "unpack_arm_64.h",
    ],
)
