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
    name = "static_assert",
    hdrs = ["static_assert.h"],
)

iree_runtime_cc_library(
    name = "exported_bits",
    hdrs = ["exported_bits.h"],
    deps = [":static_assert"],
)

ukernel_headers = [
    "common.h",
    "elementwise.h",
    "mmt4d.h",
    "pack.h",
    "query_tile_sizes.h",
    "unpack.h",
]

# :types is the type declarations used by both the entry points and the
# internal implementation functions.
iree_runtime_cc_library(
    name = "headers",
    hdrs = ukernel_headers,
    deps = [
        ":exported_bits",
        ":static_assert",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/builtins/ukernel/arch:config",
    ],
)

# Entry points.
iree_runtime_cc_library(
    name = "ukernel",
    srcs = [
        "mmt4d.c",
        "pack.c",
        "query_tile_sizes.c",
        "unpack.c",
        "elementwise_generic.c",
        "elementwise_impl.c.inc",
        "mmt4d_tile.c",
        "pack_tile.c",
        "mmt4d_tile.h",
        "pack_tile.h",
        "unpack_tile.c",
        "unpack_tile.h",
    ] + ukernel_headers,
    hdrs = ["api.h"],
    deps = [
        ":exported_bits",
        ":headers",
        ":static_assert",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/builtins/ukernel/arch:config",
        "//runtime/src/iree/builtins/ukernel/arch:ukernel_arch",
    ],
)
