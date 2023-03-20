# Copyright 2021 The IREE Authors
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
    name = "Analysis",
    srcs = [
        "Partitioning.cpp",
        "Partitioning/ReferencePartitioning.cpp",
        "ResourceHazards.cpp",
        "ResourceUsage.cpp",
    ],
    hdrs = [
        "Partitioning.h",
        "ResourceHazards.h",
        "ResourceUsage.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Dialect/Stream/IR",
        "//compiler/src/iree/compiler/Dialect/Util/Analysis",
        "//compiler/src/iree/compiler/Dialect/Util/Analysis/DFX",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
