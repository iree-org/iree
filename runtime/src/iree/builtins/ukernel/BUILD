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

# :types is the type declarations used by both the entry points and the
# internal implementation functions.
iree_runtime_cc_library(
    name = "common",
    hdrs = [
        "common.h",
        "mmt4d_types.h",
        "pack_types.h",
        "query_tile_sizes_types.h",
        "unpack_types.h",
    ],
    deps = [
        ":exported_bits",
        ":static_assert",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/builtins/ukernel/arch:config",
    ],
)

# :generic contains non-architecture-specific implementations.
iree_runtime_cc_library(
    name = "generic",
    srcs = [
        "mmt4d_generic.c",
        "pack_generic.c",
    ],
    hdrs = [
        "mmt4d_generic.h",
        "pack_generic.h",
    ],
    deps = [
        ":common",
    ],
)

# elementwise code is structured differently from other kernels. In fact it's
# profoundly different: it carries its own custom shims. For now, we keep it
# separate from the rest.
iree_runtime_cc_library(
    name = "elementwise",
    srcs = [
        "elementwise_generic.c",
        "elementwise_impl.c.inc",
    ],
    hdrs = [
        "elementwise.h",
    ],
    deps = [
        ":common",
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
    ],
    hdrs = [
        "elementwise.h",
        "mmt4d.h",
        "pack.h",
        "query_tile_sizes.h",
        "unpack.h",
    ],
    deps = [
        ":common",
        ":elementwise",
        ":generic",
        "//runtime/src/iree/builtins/ukernel/arch:ukernel_arch",
    ],
)
