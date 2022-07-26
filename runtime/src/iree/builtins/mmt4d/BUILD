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
    name = "mmt4d",
    srcs = [
        "mmt4d_arm_64.c",
        "mmt4d_generic.c",
    ],
    hdrs = [
        "mmt4d.h",
        "mmt4d_generic.h",
    ],
    copts = [
        # Placeholder for a real flag.
        "-DIREE_MMT4D_PLATFORM_EXAMPLE_FLAG=1",
    ],
    defines = [
        "IREE_HAVE_MMT4D_BUILTINS=1",
    ],
    deps = [
        "//runtime/src/iree/base:core_headers",
    ],
)
