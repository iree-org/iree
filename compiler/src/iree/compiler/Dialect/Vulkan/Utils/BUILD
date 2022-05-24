# Copyright 2019 The IREE Authors
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
    name = "Utils",
    srcs = [
        "TargetEnvironment.cpp",
        "TargetTriple.cpp",
    ],
    hdrs = [
        "TargetEnvironment.h",
        "TargetTriple.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Dialect/Vulkan/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:Support",
    ],
)
