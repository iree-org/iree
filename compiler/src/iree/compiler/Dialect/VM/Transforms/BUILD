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
        "Conversion.cpp",
        "DeduplicateRodata.cpp",
        "DropEmptyModuleInitializers.cpp",
        "GlobalInitialization.cpp",
        "HoistInlinedRodata.cpp",
        "OrdinalAllocation.cpp",
        "Passes.cpp",
        "SinkDefiningOps.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Dialect/Util/Conversion",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Dialect/VM/Conversion",
        "//compiler/src/iree/compiler/Dialect/VM/Conversion/MathToVM",
        "//compiler/src/iree/compiler/Dialect/VM/Conversion/MemRefToVM",
        "//compiler/src/iree/compiler/Dialect/VM/Conversion/StandardToVM",
        "//compiler/src/iree/compiler/Dialect/VM/Conversion/UtilToVM",
        "//compiler/src/iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithmeticTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
