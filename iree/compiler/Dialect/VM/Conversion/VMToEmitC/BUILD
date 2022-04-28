# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT IREE_ENABLE_EMITC)
  return()
endif()
""",
)

cc_library(
    name = "VMToEmitC",
    srcs = [
        "ConvertVMToEmitC.cpp",
        "DropExcludedExports.cpp",
        "EmitCTypeConverter.cpp",
    ],
    hdrs = [
        "ConvertVMToEmitC.h",
        "DropExcludedExports.h",
        "EmitCTypeConverter.h",
        "VMAnalysis.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Util/Conversion",
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/VM/Analysis",
        "//iree/compiler/Dialect/VM/IR",
        "//iree/compiler/Dialect/VM/Utils:CallingConvention",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ControlFlowOps",
        "@llvm-project//mlir:EmitC",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
