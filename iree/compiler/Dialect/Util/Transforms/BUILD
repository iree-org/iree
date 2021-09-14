# Copyright 2019 The IREE Authors
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
    name = "Transforms",
    srcs = [
        "ApplyPatterns.cpp",
        "CombineInitializers.cpp",
        "DropCompilerHints.cpp",
        "FoldGlobals.cpp",
        "Patterns.cpp",
        "SimplifyGlobalAccesses.cpp",
    ],
    hdrs = [
        "Passes.h",
        "Patterns.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Util/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
