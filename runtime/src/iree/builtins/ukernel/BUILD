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

# :types is the type declarations used by both the entry points and the
# internal implementation functions.
iree_runtime_cc_library(
    name = "types",
    hdrs = [
        "common.h",
        "mmt4d_types.h",
    ],
    deps = [
        "//runtime/src/iree/base:core_headers",
    ],
)

# :generic contains non-architecture-specific implementations.
iree_runtime_cc_library(
    name = "generic",
    srcs = [
        "mmt4d_select_tile_generic.c",
    ],
    hdrs = [
        "mmt4d_select_tile_generic.h",
    ],
    deps = [
        ":types",
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
        ":types",
    ],
)

# Entry points.
iree_runtime_cc_library(
    name = "ukernel",
    srcs = [
        "mmt4d.c",
    ],
    hdrs = [
        "elementwise.h",
        "mmt4d.h",
    ],
    deps = [
        ":elementwise",
        ":generic",
        ":types",
        "//runtime/src/iree/builtins/ukernel/arm_64:mmt4d_select_tile_arm_64",
    ],
)
