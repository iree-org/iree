# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

copy_file(
    name = "gen_config",
    src = "config.h.bazel-generic",
    out = "config.h",
)

iree_runtime_cc_library(
    name = "config",
    hdrs = ["config.h"],
)

# :types is the type declarations used by both the entry points and the
# internal implementation functions.
iree_runtime_cc_library(
    name = "ukernel_arch",
    srcs = [
        "mmt4d_arch.c",
        "pack_arch.c",
        "query_tile_sizes_arch.c",
    ],
    hdrs = [
        "mmt4d_arch.h",
        "pack_arch.h",
        "query_tile_sizes_arch.h",
    ],
    deps = [
        "//runtime/src/iree/builtins/ukernel:headers",
    ],
)
