# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")

iree_compiler_cc_library(
    name = "PluginAPI",
    hdrs = ["Registration.h"],
)

iree_compiler_cc_library(
    name = "PluginManager",
    srcs = [
        "PluginManager.cpp",
    ],
    deps = [
        ":PluginAPI",
        "//compiler/src/iree/compiler/PluginAPI/Config:StaticLinkedPlugins",
        "//compiler/src/iree/compiler/Utils",
    ],
)
