# Copyright 2020 The IREE Authors
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
if(NOT "${IREE_TARGET_BACKEND_VULKAN-SPIRV}" AND
   NOT "${IREE_TARGET_BACKEND_METAL-SPIRV}")
  return()
endif()
""",
)

cc_library(
    name = "SPIRVCommon",
    srcs = [
        "SPIRVTarget.cpp",
    ],
    hdrs = [
        "SPIRVTarget.h",
    ],
    deps = [
        "//iree/compiler/Conversion:PassHeaders",
        "//iree/compiler/Conversion/CodegenUtils",
        "//iree/compiler/Conversion/Common",
        "//iree/compiler/Conversion/LinalgToSPIRV",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SPIRVConversion",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:Support",
    ],
)
