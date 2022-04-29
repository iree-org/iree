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
    name = "TensorToFlow",
    srcs = [
        "ConvertTensorToFlow.cpp",
    ],
    hdrs = [
        "ConvertTensorToFlow.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Flow/IR",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:TensorDialect",
    ],
)
