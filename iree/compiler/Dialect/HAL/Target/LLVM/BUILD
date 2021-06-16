# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content", "platform_trampoline_deps")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT "${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT}" AND NOT "${IREE_TARGET_BACKEND_WASM-LLVM-AOT}")
  return()
endif()
""",
)

cc_library(
    name = "LLVM",
    srcs = [
        "LLVMAOTTarget.cpp",
        "LibraryBuilder.cpp",
    ],
    hdrs = [
        "LLVMAOTTarget.h",
        "LibraryBuilder.h",
    ],
    deps = [
        ":LLVMIRPasses",
        ":LLVMTargetOptions",
        ":LinkerTool",
        ":StaticLibraryGenerator",
        "//iree/base/internal:flatcc",
        "//iree/compiler/Conversion:PassHeaders",
        "//iree/compiler/Conversion/CodegenUtils",
        "//iree/compiler/Conversion/Common",
        "//iree/compiler/Conversion/LinalgToLLVM",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Utils",
        "//iree/schemas:dylib_executable_def_c_fbs",
        "@llvm-project//llvm:AArch64AsmParser",
        "@llvm-project//llvm:AArch64CodeGen",
        "@llvm-project//llvm:ARMAsmParser",
        "@llvm-project//llvm:ARMCodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:RISCVAsmParser",
        "@llvm-project//llvm:RISCVCodeGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:WebAssemblyAsmParser",
        "@llvm-project//llvm:WebAssemblyCodeGen",
        "@llvm-project//llvm:X86AsmParser",
        "@llvm-project//llvm:X86CodeGen",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)

cc_library(
    name = "LLVMIRPasses",
    srcs = [
        "LLVMIRPasses.cpp",
    ],
    hdrs = [
        "LLVMIRPasses.h",
    ],
    deps = [
        ":LLVMTargetOptions",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Instrumentation",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "LLVMTargetOptions",
    srcs = [
        "LLVMTargetOptions.cpp",
    ],
    hdrs = [
        "LLVMTargetOptions.h",
    ],
    deps = [
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
    ],
)

cc_library(
    name = "LinkerTool",
    srcs = ["LinkerTool.cpp"],
    hdrs = ["LinkerTool.h"],
    deps = platform_trampoline_deps("LinkerTools", "compiler/Dialect/HAL/Target/LLVM") + [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "LinkerTool_hdrs",
    hdrs = ["LinkerTool.h"],
    deps = [
        ":LLVMTargetOptions",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "StaticLibraryGenerator",
    srcs = [
        "StaticLibraryGenerator.cpp",
    ],
    hdrs = [
        "StaticLibraryGenerator.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)
