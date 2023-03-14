# Copyright 2021 The IREE Authors
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
if(NOT IREE_TARGET_BACKEND_ROCM)
  return()
endif()
""",
)

iree_compiler_cc_library(
    name = "ROCM",
    srcs = [
        "ROCMTarget.cpp",
        "ROCMTargetUtils.cpp",
    ],
    hdrs = [
        "ROCMTarget.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Codegen:PassHeaders",
        "//compiler/src/iree/compiler/Codegen/Dialect:IREECodegenDialect",
        "//compiler/src/iree/compiler/Codegen/LLVMGPU",
        "//compiler/src/iree/compiler/Dialect/HAL/Target",
        "//compiler/src/iree/compiler/Utils",
        "//runtime/src/iree/schemas:rocm_executable_def_c_fbs",
        "@llvm-project//llvm:AMDGPUCodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)
