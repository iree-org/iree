# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_compiler_cc_library(
    name = "Embed",
    srcs = [
        "Diagnostics.cpp",
        "Embed.cpp",
    ],
    hdrs = [
        "Diagnostics.h",
    ],
    deps = [
        "//compiler/bindings/c:headers",
        "//compiler/src/iree/compiler/ConstEval",
        "//compiler/src/iree/compiler/Dialect/VM/Target:init_targets",
        "//compiler/src/iree/compiler/Dialect/VM/Target/Bytecode",
        "//compiler/src/iree/compiler/Dialect/VM/Target/C",
        "//compiler/src/iree/compiler/Pipelines",
        "//compiler/src/iree/compiler/Tools:init_llvmir_translations",
        "//compiler/src/iree/compiler/Tools:init_passes_and_dialects",
        "//compiler/src/iree/compiler/Tools:init_targets",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

iree_compiler_cc_library(
    name = "IREECompileToolEntryPoint",
    srcs = [
        "IREECompileToolEntryPoint.cpp",
    ],
    deps = [
        "//compiler/bindings/c:headers",
        "//compiler/src/iree/compiler/Tools:iree_compile_lib",
    ],
)

iree_compiler_cc_library(
    name = "IREEMLIRLSPServerToolEntryPoint",
    srcs = [
        "IREEMLIRLSPServerToolEntryPoint.cpp",
    ],
    deps = [
        "//compiler/bindings/c:headers",
        "//compiler/src/iree/compiler/Tools:init_passes_and_dialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirLspServerLib",
        "@llvm-project//mlir:Support",
    ],
)

iree_compiler_cc_library(
    name = "IREEOptToolEntryPoint",
    srcs = [
        "IREEOptToolEntryPoint.cpp",
    ],
    deps = [
        "//compiler/bindings/c:headers",
        "//compiler/src/iree/compiler/Tools:init_passes_and_dialects",
        "//compiler/src/iree/compiler/Tools:init_targets",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Support",
    ],
)

iree_compiler_cc_library(
    name = "LLDToolEntryPoint",
    srcs = [
        "LLDToolEntryPoint.cpp",
    ],
    deps = [
        "//compiler/bindings/c:headers",
        "@llvm-project//lld:COFF",
        "@llvm-project//lld:Common",
        "@llvm-project//lld:ELF",
        "@llvm-project//lld:MachO",
        "@llvm-project//lld:Wasm",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

iree_compiler_cc_library(
    name = "MLIRInterop",
    srcs = [
        "MLIRInterop.cpp",
    ],
    deps = [
        "//compiler/src/iree/compiler/API2:MLIRInteropHeaders",
        "//compiler/src/iree/compiler/ConstEval",
        "//compiler/src/iree/compiler/Dialect/VM/IR",
        "//compiler/src/iree/compiler/Dialect/VM/Target:init_targets",
        "//compiler/src/iree/compiler/Dialect/VM/Target/Bytecode",
        "//compiler/src/iree/compiler/Pipelines",
        "//compiler/src/iree/compiler/Tools:init_llvmir_translations",
        "//compiler/src/iree/compiler/Tools:init_passes_and_dialects",
        "//compiler/src/iree/compiler/Tools:init_targets",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
    ],
)
