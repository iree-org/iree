# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT IREE_TARGET_BACKEND_VULKAN_SPIRV)
  return()
endif()
""",
)

iree_compiler_cc_library(
    name = "VulkanSPIRV",
    srcs = [
        "VulkanSPIRVTarget.cpp",
    ],
    hdrs = [
        "VulkanSPIRVTarget.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Codegen:PassHeaders",
        "//compiler/src/iree/compiler/Codegen/Common",
        "//compiler/src/iree/compiler/Codegen/Dialect:IREECodegenDialect",
        "//compiler/src/iree/compiler/Codegen/SPIRV",
        "//compiler/src/iree/compiler/Codegen/Utils",
        "//compiler/src/iree/compiler/Dialect/HAL/Target",
        "//compiler/src/iree/compiler/Dialect/Vulkan/IR",
        "//compiler/src/iree/compiler/Dialect/Vulkan/Utils",
        "//compiler/src/iree/compiler/Utils",
        "//runtime/src/iree/schemas:spirv_executable_def_c_fbs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVModuleCombiner",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:Support",
    ],
)
