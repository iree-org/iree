# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library", "iree_gentbl_cc_library")

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
        "IPO.cpp",
        "PassDetail.h",
        "Passes.cpp",
        "Patterns.cpp",
        "PropagateSubranges.cpp",
        "SimplifyGlobalAccesses.cpp",
        "StripDebugOps.cpp",
        "TestConversion.cpp",
        "TestFloatRangeAnalysis.cpp",
    ],
    hdrs = [
        "Passes.h",
        "Passes.h.inc",
        "Patterns.h",
    ],
    deps = [
        ":PassesIncGen",
        "//compiler/src/iree/compiler/Dialect/Util/Analysis",
        "//compiler/src/iree/compiler/Dialect/Util/Analysis/Attributes",
        "//compiler/src/iree/compiler/Dialect/Util/Analysis/Constant",
        "//compiler/src/iree/compiler/Dialect/Util/Analysis/DFX",
        "//compiler/src/iree/compiler/Dialect/Util/Conversion",
        "//compiler/src/iree/compiler/Dialect/Util/Conversion/MemRefToUtil",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

iree_gentbl_cc_library(
    name = "PassesIncGen",
    tbl_outs = [
        (
            ["--gen-pass-decls"],
            "Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
