# Copyright 2021 The IREE Authors
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
if(NOT "${IREE_TARGET_BACKEND_ROCM}")
  return()
endif()
""",
)

cc_library(
    name = "ROCM",
    srcs = [
        "ROCMTarget.cpp",
        "ROCMTargetUtils.cpp",
    ],
    hdrs = [
        "ROCMTarget.h",
    ],
    deps = [
        "//iree/base/internal:flatcc",
        "//iree/compiler/Conversion/LinalgToLLVMGPU",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Utils",
        "//iree/schemas:rocm_executable_def_c_fbs",
        "@llvm-project//llvm:AMDGPUCodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
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
