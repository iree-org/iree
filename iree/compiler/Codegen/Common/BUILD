# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl_cc_library(
    name = "FoldTensorExtractOpIncGen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "FoldTensorExtractOp.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FoldTensorExtractOp.td",
    deps = [
        "@llvm-project//mlir:MemRefOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

cc_library(
    name = "Common",
    srcs = [
        "BufferizationAnalysis.cpp",
        "CleanupBufferAllocViewPass.cpp",
        "ConvertToDestinationPassingStylePass.cpp",
        "DemoteF32ToF16.cpp",
        "FlattenMemRefSubspanPass.cpp",
        "FoldAffineMinInDistributedLoops.cpp",
        "FoldTensorExtractOpPass.cpp",
        "ForOpCanonicalizationPass.cpp",
        "IREEComprehensiveBufferizePass.cpp",
        "LinalgBufferizePass.cpp",
        "OptimizeVectorTransferPass.cpp",
        "RemoveTrivialLoops.cpp",
        "SetNumWorkgroupsPass.cpp",
        "VectorizeConv.cpp",
        "VectorizeMMT4d.cpp",
    ],
    hdrs = [
        "BufferizationAnalysis.h",
    ],
    deps = [
        "//iree/compiler/Codegen:PassHeaders",
        "//iree/compiler/Codegen/Common:FoldTensorExtractOpIncGen",
        "//iree/compiler/Codegen/Dialect:IREECodegenDialect",
        "//iree/compiler/Codegen/Transforms",
        "//iree/compiler/Codegen/Utils",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/Util/IR",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "//llvm-external-projects/iree-dialects:IREELinalgExtTransforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:BufferizableOpInterface",
        "@llvm-project//mlir:CFGTransforms",
        "@llvm-project//mlir:ComprehensiveBufferize",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgBufferizableOpInterfaceImpl",
        "@llvm-project//mlir:LinalgInterfaces",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorBufferizableOpInterfaceImpl",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorBufferizableOpInterfaceImpl",
        "@llvm-project//mlir:VectorOps",
    ],
)
