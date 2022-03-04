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
        "TypeConverter.cpp",
    ],
    hdrs = [
        "ConversionDialectInterface.h",
        "ConversionTarget.h",
        "TypeConverter.h",
    ],
    deps = [
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/HAL/Utils",
        "//iree/compiler/Dialect/Util/IR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "Passes",
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//iree/compiler/Dialect/HAL/Conversion/HALToVM",
        "@llvm-project//mlir:Pass",
    ],
)
