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
    name = "Conversion",
    srcs = [
        "ConversionTarget.cpp",
        "ImportUtils.cpp",
        "TargetOptions.cpp",
        "TypeConverter.cpp",
    ],
    hdrs = [
        "ConversionDialectInterface.h",
        "ConversionTarget.h",
        "ImportUtils.h",
        "TargetOptions.h",
        "TypeConverter.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)
