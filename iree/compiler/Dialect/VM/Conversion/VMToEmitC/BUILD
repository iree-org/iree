# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

iree_cmake_extra_content(content = """
if(NOT ${IREE_ENABLE_EMITC})
  return()
endif()
""")

cc_library(
    name = "VMToEmitC",
    srcs = [
        "ConvertVMToEmitC.cpp",
        "DropExcludedExports.cpp",
    ],
    hdrs = [
        "ConvertVMToEmitC.h",
        "DropExcludedExports.h",
        "EmitCTypeConverter.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Util/Conversion",
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/VM/Analysis",
        "//iree/compiler/Dialect/VM/IR",
        "//iree/compiler/Dialect/VM/Utils:CallingConvention",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:EmitC",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)
