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
    name = "StandardToVMVX",
    srcs = [
        "ConvertStandardToVMVX.cpp",
    ],
    hdrs = [
        "ConvertStandardToVMVX.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Modules/VMVX/IR",
        "//iree/compiler/Dialect/Modules/VMVX/IR:VMVXDialect",
        "//iree/compiler/Dialect/Util/IR",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
