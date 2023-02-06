# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.bzl", "iree_pjrt_cc_library", "iree_pjrt_plugin_dylib")

iree_pjrt_cc_library(
    name = "client",
    srcs = [
        "client.cc",
    ],
    hdrs = [
        "client.h",
    ],
    deps = [
        "//iree/integrations/pjrt/common:impl",
        "@iree_core//runtime/src/iree/hal/drivers/local_task:task_driver",
        "@iree_core//runtime/src/iree/hal/local",
        "@iree_core//runtime/src/iree/hal/local:executable_loader",
        "@iree_core//runtime/src/iree/hal/local/loaders/registration",
        "@iree_core//runtime/src/iree/task:api",
    ],
)

iree_pjrt_plugin_dylib(
    name = "pjrt_plugin_iree_cpu",
    srcs = [
        "dylib_entry_point.cc",
    ],
    deps = [
        ":client",
    ],
)
