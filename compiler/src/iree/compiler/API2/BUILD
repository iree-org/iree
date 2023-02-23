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

exports_files([
    "api_exports.c",
    "api_exports.def",
    "api_exports.ld",
    "api_exports.lst",
])

# All headers for APIs exported by the library.
iree_compiler_cc_library(
    name = "MLIRInteropHeaders",
    hdrs = [
        "MLIRInterop.h",
    ],
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

iree_compiler_cc_library(
    name = "StaticImpl",
    deps = [
        "//compiler/src/iree/compiler/API2/Internal:Embed",
        "//compiler/src/iree/compiler/API2/Internal:IREECompileToolEntryPoint",
        "//compiler/src/iree/compiler/API2/Internal:IREEMLIRLSPServerToolEntryPoint",
        "//compiler/src/iree/compiler/API2/Internal:IREEOptToolEntryPoint",
        "//compiler/src/iree/compiler/API2/Internal:LLDToolEntryPoint",
        "//compiler/src/iree/compiler/API2/Internal:MLIRInterop",
        "//llvm-external-projects/iree-dialects:CAPI",
        "@llvm-project//mlir:CAPIDebug",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:CAPIInterfaces",
        "@llvm-project//mlir:CAPILinalg",
        "@llvm-project//mlir:CAPIPDL",
        "@llvm-project//mlir:CAPITransformDialect",
    ],
)

# Bazel does not have a well-defined mechanism for managing
# shared libraries in a cross-platform way. Therefore, we
# only support static linking of tools. Separately, we do
# build an actual shared library for platforms that support
# it, but this is only done to support stub use-cases.
alias(
    name = "Impl",
    actual = ":StaticImpl",
)
