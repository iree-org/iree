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
    name = "LinalgToLLVMGPU",
    srcs = [
        "ConvertToLLVM.cpp",
        "ConvertToNVVM.cpp",
        "ConvertToROCDL.cpp",
        "KernelConfig.cpp",
        "LowerExecutableTargetPass.cpp",
        "Passes.cpp",
        "RemoveTrivialLoops.cpp",
        "TileAndDistribute.cpp",
        "VectorizationPass.cpp",
    ],
    hdrs = [
        "ConvertToLLVM.h",
        "KernelConfig.h",
    ],
    deps = [
        "//iree/compiler/Conversion:PassHeaders",
        "//iree/compiler/Conversion/Common",
        "//iree/compiler/Conversion/Transforms",
        "//iree/compiler/Conversion/Utils",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/Transforms",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUToROCDLTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:StandardToSPIRV",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorToSCF",
        "@mlir-hlo//:hlo",
        "@mlir-hlo//:legalize_to_linalg",
    ],
)
