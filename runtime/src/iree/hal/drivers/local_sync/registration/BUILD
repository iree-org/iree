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
    name = "registration",
    srcs = ["driver_module.c"],
    hdrs = ["driver_module.h"],
    defines = [
        "IREE_HAVE_HAL_LOCAL_SYNC_DRIVER_MODULE=1",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/drivers/local_sync:sync_driver",
        "//runtime/src/iree/hal/local/loaders/registration",
    ],
)
