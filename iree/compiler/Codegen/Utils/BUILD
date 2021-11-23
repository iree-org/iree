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
        "Utils.cpp",
    ],
    hdrs = [
        "MarkerUtils.h",
        "Utils.h",
    ],
    deps = [
        "//iree/compiler/Codegen/Interfaces:ProcessorOpInterfaces",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/IR",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Support",
    ],
)
