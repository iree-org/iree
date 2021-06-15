# Copyright 2019 The IREE Authors
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
if(NOT "${IREE_TARGET_BACKEND_VULKAN-SPIRV}")
  return()
endif()
""",
)

cc_library(
    name = "VulkanSPIRV",
    srcs = [
        "VulkanSPIRVTarget.cpp",
    ],
    hdrs = [
        "VulkanSPIRVTarget.h",
    ],
    deps = [
        "//iree/base/internal:flatcc",
        "//iree/compiler/Conversion:PassHeaders",
        "//iree/compiler/Conversion/Common",
        "//iree/compiler/Conversion/LinalgToSPIRV",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Dialect/HAL/Target/SPIRVCommon",
        "//iree/compiler/Dialect/Vulkan/IR",
        "//iree/compiler/Dialect/Vulkan/Utils",
        "//iree/compiler/Utils",
        "//iree/schemas:spirv_executable_def_c_fbs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:VectorOps",
    ],
)
