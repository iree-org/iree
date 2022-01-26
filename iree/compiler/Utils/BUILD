# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Utilities for working with IREE MLIR types.

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Utils",
    srcs = [
        "ConversionUtils.cpp",
        "CustomKernelsTargetInfo.cpp",
        "FlatbufferUtils.cpp",
        "GraphUtils.cpp",
        "ModuleUtils.cpp",
        "OptionUtils.cpp",
        "PassUtils.cpp",
        "StringUtils.cpp",
        "TracingUtils.cpp",
    ],
    hdrs = [
        "ConversionUtils.h",
        "CustomKernelsTargetInfo.h",
        "FlatbufferUtils.h",
        "GraphUtils.h",
        "IndexSet.h",
        "ModuleUtils.h",
        "OptionUtils.h",
        "PassUtils.h",
        "PatternUtils.h",
        "StringUtils.h",
        "TracingUtils.h",
    ],
    deps = [
        "//iree/base:tracing",
        "//iree/base/internal/flatcc:building",
        "//iree/base/internal/flatcc:debugging",
        "//iree/base/internal/flatcc:parsing",
        "//iree/compiler/Dialect/Util/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
