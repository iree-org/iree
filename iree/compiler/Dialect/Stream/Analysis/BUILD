# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Analysis",
    srcs = [
        "Partitioning.cpp",
        "Partitioning/ReferencePartitioning.cpp",
        "ResourceUsage.cpp",
    ],
    hdrs = [
        "Partitioning.h",
        "ResourceUsage.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Stream/IR",
        "//iree/compiler/Dialect/Util/Analysis",
        "//iree/compiler/Dialect/Util/Analysis/DFX",
        "//iree/compiler/Dialect/Util/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
