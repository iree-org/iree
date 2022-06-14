# Copyright 2021 The IREE Authors
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
        "Passes.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Codegen:PassHeaders",
        "//compiler/src/iree/compiler/Codegen/Common",
        "//compiler/src/iree/compiler/Codegen/LLVMCPU",
        "//compiler/src/iree/compiler/Dialect/HAL/IR:HALDialect",
        "//compiler/src/iree/compiler/Dialect/HAL/Transforms",
        "//compiler/src/iree/compiler/Dialect/Modules/VMVX/Conversion/HALToVMVX",
        "//compiler/src/iree/compiler/Dialect/Modules/VMVX/Conversion/StandardToVMVX",
        "//compiler/src/iree/compiler/Dialect/Modules/VMVX/IR",
        "//compiler/src/iree/compiler/Dialect/Modules/VMVX/IR:VMVXDialect",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Dialect/Util/Transforms",
        "//compiler/src/iree/compiler/Dialect/VM/IR",
        "//llvm-external-projects/iree-dialects:IREELinalgExtPasses",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:ArithmeticTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TosaToArith",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorToSCF",
    ],
)
