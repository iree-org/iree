# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Utilities for working with IREE MLIR types.

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Utils",
    srcs = [
        "MarkerUtils.cpp",
        "TransformUtils.cpp",
        "Utils.cpp",
    ],
    hdrs = [
        "MarkerUtils.h",
        "TransformUtils.h",
        "Utils.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/HAL/IR:HALDialect",
        "//iree/compiler/Dialect/HAL/Utils",
        "//iree/compiler/Dialect/Shape/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorToSCF",
    ],
)
