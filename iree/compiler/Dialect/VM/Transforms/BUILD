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
        "Conversion.cpp",
        "DeduplicateRodata.cpp",
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
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Util/Conversion:PreserveCompilerHints",
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/VM/Conversion",
        "//iree/compiler/Dialect/VM/Conversion/MathToVM",
        "//iree/compiler/Dialect/VM/Conversion/MemRefToVM",
        "//iree/compiler/Dialect/VM/Conversion/StandardToVM",
        "//iree/compiler/Dialect/VM/Conversion/UtilToVM",
        "//iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
