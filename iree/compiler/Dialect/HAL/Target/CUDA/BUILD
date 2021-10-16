# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT "${IREE_TARGET_BACKEND_CUDA}")
  return()
endif()
""",
)

c_embed_data(
    name = "cuda_libdevice",
    srcs = ["@cuda//:libdevice_root"],
    c_file_output = "libdevice.c",
    flatten = True,
    h_file_output = "libdevice.h",
)

cc_library(
    name = "CUDA",
    srcs = [
        "CUDATarget.cpp",
        "NoLoopUnrollPass.cpp",
    ],
    hdrs = [
        "CUDATarget.h",
        "LLVMPasses.h",
    ],
    deps = [
        ":cuda_libdevice",
        "//iree/compiler/Codegen:PassHeaders",
        "//iree/compiler/Codegen/LLVMGPU",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Utils",
        "//iree/schemas:cuda_executable_def_c_fbs",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:NVPTXCodeGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)
