# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_compiler_cc_library(
    name = "HALExecutable",
    srcs = ["HALExecutable.cpp"],
    hdrs = ["HALExecutable.h"],
    deps = [
        "//compiler/src/iree/compiler/Dialect/HAL/Target",
        "//compiler/src/iree/compiler/Dialect/HAL/Transforms",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)

iree_compiler_cc_library(
    name = "IREEVM",
    srcs = ["IREEVM.cpp"],
    hdrs = ["IREEVM.h"],
    deps = [
        "//compiler/src/iree/compiler/ConstEval",
        "//compiler/src/iree/compiler/Dialect/Flow/IR",
        "//compiler/src/iree/compiler/Dialect/HAL/Target",
        "//compiler/src/iree/compiler/Dialect/VM/Target/C",
        "//compiler/src/iree/compiler/Pipelines",
        "//compiler/src/iree/compiler/Pipelines:Options",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Translation",
    ],
)
