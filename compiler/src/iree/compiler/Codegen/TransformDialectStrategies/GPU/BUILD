# Copyright 2023 The IREE Authors
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
    name = "TransformDialectStrategies",
    srcs = [
        "Common.cpp",
        "SmallReductionStrategy.cpp",
        "StagedReductionStrategy.cpp",
    ],
    hdrs = [
        "AbstractReductionStrategy.h",
        "Common.h",
        "SmallReductionStrategy.h",
        "StagedReductionStrategy.h",
    ],
    deps = [
        # Dialects
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "//llvm-external-projects/iree-dialects:IREELinalgExtTransformOps",
        "//llvm-external-projects/iree-dialects:IREELinalgTransformDialect",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:PDLDialect",
        "@llvm-project//mlir:PDLInterpDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:VectorDialect",
        # IR
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        # Other Stuff
        "//compiler/src/iree/compiler/Codegen:PassHeaders",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:DialectUtils",
        # TransformDialectStrategies
        "//compiler/src/iree/compiler/Codegen/TransformDialectStrategies/Common:TransformDialectStrategies",
        # TransformExtensions
        "//compiler/src/iree/compiler/Codegen/Common/TransformExtensions:CommonExtensions",
        "//compiler/src/iree/compiler/Codegen/LLVMGPU/TransformExtensions:LLVMGPUExtensions",
        "@llvm-project//mlir:LinalgTransformOps",
        # TransformMatchers and other stuff
        "//llvm-external-projects/iree-dialects:IREEDialectsTransforms",
    ],
)
