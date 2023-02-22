# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_compiler_cc_library(
    name = "LinkerTools_internal",
    srcs = [
        "AndroidLinkerTool.cpp",
        "EmbeddedLinkerTool.cpp",
        "LinkerTools.cpp",
        "UnixLinkerTool.cpp",
        "WasmLinkerTool.cpp",
        "WindowsLinkerTool.cpp",
    ],
    deps = [
        "//compiler/src/iree/compiler/Dialect/HAL/Target/LLVM:LinkerTool_hdrs",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:Support",
    ],
)
