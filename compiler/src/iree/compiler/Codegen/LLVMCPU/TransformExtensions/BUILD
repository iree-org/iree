# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library", "iree_gentbl_cc_library", "iree_td_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "LLVMCPUExtensionsOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:TransformDialectTdFiles",
    ],
)

iree_gentbl_cc_library(
    name = "LLVMCPUExtensionsOpGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "LLVMCPUExtensionsOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "LLVMCPUExtensionsOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LLVMCPUExtensionsOps.td",
    deps = [":td_files"],
)

iree_compiler_cc_library(
    name = "LLVMCPUExtensions",
    srcs = [
        "LLVMCPUExtensions.cpp",
        "LLVMCPUExtensionsOps.cpp.inc",
    ],
    hdrs = [
        "LLVMCPUExtensions.h",
        "LLVMCPUExtensionsOps.h.inc",
    ],
    deps = [
        ":LLVMCPUExtensionsOpGen",
        "//llvm-external-projects/iree-dialects:IREEDialectsTransforms",
        "//llvm-external-projects/iree-dialects:IREELinalgTransformDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TransformDialect",
    ],
)
