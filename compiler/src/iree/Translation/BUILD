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
    name = "HALExecutable",
    srcs = ["HALExecutable.cpp"],
    hdrs = ["HALExecutable.h"],
    deps = [
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Dialect/HAL/Transforms",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)

cc_library(
    name = "IREEVM",
    srcs = ["IREEVM.cpp"],
    hdrs = ["IREEVM.h"],
    deps = [
        "//iree/compiler/ConstEval",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Dialect/VM/Target/C",
        "//iree/compiler/Pipelines",
        "//iree/compiler/Pipelines:Options",
        "//iree/compiler/Utils",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Translation",
    ],
)
