# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_compiler_cc_library(
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
        "//compiler/src/iree/compiler/Dialect/HAL/IR",
        "//compiler/src/iree/compiler/Dialect/HAL/Utils",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

iree_compiler_cc_library(
    name = "Passes",
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Dialect/HAL/Conversion/HALToVM",
        "@llvm-project//mlir:Pass",
    ],
)
