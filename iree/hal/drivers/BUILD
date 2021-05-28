# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "IREE_CUDA_DEPS")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "drivers",
    srcs = ["init.c"],
    hdrs = ["init.h"],
    deps = [
        "//iree/base",
        "//iree/base:tracing",
    ] + [
        # TODO(*): select() and only pull in based on build configuration.
        "//iree/hal/dylib/registration",
        "//iree/hal/dylib/registration:sync",
        "//iree/hal/vmvx/registration",
        "//iree/hal/vulkan/registration",
    ] + IREE_CUDA_DEPS,
)
