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
    name = "Transforms",
    srcs = [
        "ApplyPatterns.cpp",
        "CombineInitializers.cpp",
        "ConvertPrimitiveType.cpp",
        "DropCompilerHints.cpp",
        "FixedPointIterator.cpp",
        "FoldGlobals.cpp",
        "FuseGlobals.cpp",
        "HoistIntoGlobals.cpp",
        "Patterns.cpp",
        "SimplifyGlobalAccesses.cpp",
        "StripDebugOps.cpp",
        "TestFloatRangeAnalysis.cpp",
    ],
    hdrs = [
        "Passes.h",
        "Patterns.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Dialect/Util/Analysis",
        "//compiler/src/iree/compiler/Dialect/Util/Analysis/Attributes",
        "//compiler/src/iree/compiler/Dialect/Util/Analysis/Constant",
        "//compiler/src/iree/compiler/Dialect/Util/Analysis/DFX",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
