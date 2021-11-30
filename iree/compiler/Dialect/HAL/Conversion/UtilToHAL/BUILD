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
    name = "UtilToHAL",
    srcs = [
        "ConvertUtilToHAL.cpp",
    ],
    hdrs = [
        "ConvertUtilToHAL.h",
    ],
    deps = [
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/Util/Conversion",
        "//iree/compiler/Dialect/Util/IR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)
