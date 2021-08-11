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
        "CleanupBufferAllocViewPass.cpp",
        "DemoteF32ToF16.cpp",
        "FlattenMemRefSubspanPass.cpp",
        "FoldTensorExtractOpPass.cpp",
        "ForOpCanonicalizationPass.cpp",
        "LinalgBufferizePass.cpp",
        "OptimizeVectorTransferPass.cpp",
        "SetNumWorkgroupsPass.cpp",
        "ShapeToLLVMConversion.cpp",
        "VectorizeConv.cpp",
        "VectorizeMMT4d.cpp",
    ],
    deps = [
        "//iree/compiler/Codegen:PassHeaders",
        "//iree/compiler/Codegen/Common:FoldTensorExtractOpIncGen",
        "//iree/compiler/Codegen/Transforms",
        "//iree/compiler/Codegen/Utils",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/LinalgExt/IR",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Util/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:CFGTransforms",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
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
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
    ],
)
