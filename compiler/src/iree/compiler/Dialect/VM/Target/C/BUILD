# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT IREE_ENABLE_EMITC)
  # If the real "C" library is enabled, it will define IREE_ENABLE_EMITC,
  # which can be used for conditionally switching.
  iree_cc_library(
    NAME Enabled
  )
  iree_cc_library(
    NAME C
  )
  iree_cc_library(
    NAME TranslateToCpp
  )
  return()
endif()
""",
)

iree_compiler_cc_library(
    name = "Enabled",
    defines = [
        "IREE_HAVE_EMITC_DIALECT",
    ],
    deps = [
        "@llvm-project//mlir:EmitC",
    ],
)

iree_compiler_cc_library(
    name = "TranslateToCpp",
    srcs = [
        "TranslateToCpp.cpp",
    ],
    hdrs = [
        "CppEmitter.h",
    ],
    deps = [
        ":Enabled",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowOps",
        "@llvm-project//mlir:EmitC",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
    ],
)

iree_compiler_cc_library(
    name = "C",
    srcs = [
        "CModuleTarget.cpp",
        "TranslationFlags.cpp",
        "TranslationRegistration.cpp",
    ],
    hdrs = [
        "CModuleTarget.h",
        "TranslationFlags.h",
    ],
    deps = [
        ":Enabled",
        ":TranslateToCpp",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Dialect/Util/Transforms",
        "//compiler/src/iree/compiler/Dialect/VM/Analysis",
        "//compiler/src/iree/compiler/Dialect/VM/Conversion/VMToEmitC",
        "//compiler/src/iree/compiler/Dialect/VM/IR",
        "//compiler/src/iree/compiler/Dialect/VM/Transforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:Translation",
    ],
)
