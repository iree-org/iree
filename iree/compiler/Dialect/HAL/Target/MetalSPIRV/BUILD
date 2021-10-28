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
if(NOT "${IREE_TARGET_BACKEND_METAL-SPIRV}")
  return()
endif()
""",
)

cc_library(
    name = "MetalSPIRV",
    srcs = ["MetalSPIRVTarget.cpp"],
    hdrs = ["MetalSPIRVTarget.h"],
    deps = [
        ":SPIRVToMSL",
        "//iree/compiler/Codegen:PassHeaders",
        "//iree/compiler/Codegen/Common",
        "//iree/compiler/Codegen/Dialect:IREECodegenDialect",
        "//iree/compiler/Codegen/SPIRV",
        "//iree/compiler/Codegen/Utils",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Utils",
        "//iree/schemas:metal_executable_def_c_fbs",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:VectorOps",
    ],
)

cc_library(
    name = "SPIRVToMSL",
    srcs = [
        "SPIRVToMSL.cpp",
    ],
    hdrs = ["SPIRVToMSL.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@spirv_cross//:spirv_cross_lib",
    ],
)
