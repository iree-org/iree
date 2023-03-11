# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_register_plugin")

cc_library(
    name = "defs",
    include = ["src"],
)

cc_library(
    name = "registration",
    srcs = [
        "src/PluginRegistration.cpp",
    ],
    deps = [
        ":defs",
        "//compiler/src/iree/compiler/PluginAPI",
    ],
)

iree_compiler_register_plugin(
    plugin_id = "example",
    target = ":registration",
)
