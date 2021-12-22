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
if(NOT "${IREE_TARGET_BACKEND_DYLIB_LLVM_AOT}" AND NOT "${IREE_TARGET_BACKEND_WASM_LLVM_AOT}")
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
        "//iree/compiler/Codegen:PassHeaders",
        "//iree/compiler/Codegen/Common",
        "//iree/compiler/Codegen/Dialect:IREECodegenDialect",
        "//iree/compiler/Codegen/LLVMCPU",
        "//iree/compiler/Codegen/Utils",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Dialect/HAL/Target/LLVM/Builtins",
        "@llvm-project//llvm:AArch64AsmParser",
        "@llvm-project//llvm:AArch64CodeGen",
        "@llvm-project//llvm:ARMAsmParser",
        "@llvm-project//llvm:ARMCodeGen",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Linker",
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
        "@llvm-project//llvm:MC",
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
