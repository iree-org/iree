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
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/Util/Transforms",
        "//iree/compiler/Dialect/VM/Analysis",
        "//iree/compiler/Dialect/VM/Conversion/VMToEmitC",
        "//iree/compiler/Dialect/VM/IR",
        "//iree/compiler/Dialect/VM/Transforms",
        "@emitc//:TranslateToCpp",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:Translation",
    ],
)
