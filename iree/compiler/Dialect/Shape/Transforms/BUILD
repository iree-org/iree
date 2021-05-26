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
        "CleanupPlaceholdersPass.cpp",
        "ConvertHLOToShapeDialectPass.cpp",
        "ConvertShapeToStandard.cpp",
        "FunctionSignatureExpansionPass.cpp",
        "HoistShapeCalculationsPass.cpp",
        "MaterializeShapeCalculations.cpp",
        "MaterializeShapeCalculationsPass.cpp",
        "TieDynamicShapesPass.cpp",
    ],
    hdrs = [
        "Passes.h",
        "Patterns.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Shape/Plugins/XLA:XlaHloShapeBuilder",
        "//iree/compiler/Dialect/Shape/Utils:TypeConversion",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@mlir-hlo//:hlo",
    ],
)
