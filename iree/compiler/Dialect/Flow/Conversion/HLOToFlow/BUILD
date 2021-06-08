# Copyright 2019 The IREE Authors
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
    name = "HLOToFlow",
    srcs = [
        "ConvertHLOToFlow.cpp",
    ],
    hdrs = [
        "ConvertHLOToFlow.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/IREE/IR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@mlir-hlo//:hlo",
    ],
)
