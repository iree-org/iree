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
    name = "UtilToVM",
    srcs = [
        "ConvertGlobalOps.cpp",
        "ConvertListOps.cpp",
        "ConvertUtilToVM.cpp",
    ],
    hdrs = [
        "ConvertUtilToVM.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/VM/Conversion",
        "//iree/compiler/Dialect/VM/IR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
