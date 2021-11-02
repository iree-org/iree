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
    name = "StandardToStream",
    srcs = [
        "ConvertConstantOps.cpp",
        "ConvertStandardToStream.cpp",
        "ConvertStructuralOps.cpp",
    ],
    hdrs = [
        "ConvertStandardToStream.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Stream/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
