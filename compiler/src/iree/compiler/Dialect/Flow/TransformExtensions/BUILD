# Copyright 2022 The IREE Authors
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
            "FlowExtensionsOps.td",
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
    name = "FlowExtensionsOpGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "FlowExtensionsOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "FlowExtensionsOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowExtensionsOps.td",
    deps = [":td_files"],
)

iree_compiler_cc_library(
    name = "FlowExtensions",
    srcs = [
        "FlowExtensions.cpp",
        "FlowExtensionsOps.cpp.inc",
    ],
    hdrs = [
        "FlowExtensions.h",
        "FlowExtensionsOps.h.inc",
    ],
    deps = [
        ":FlowExtensionsOpGen",
        "//compiler/src/iree/compiler/Dialect/Flow/IR",
        "//compiler/src/iree/compiler/Dialect/Flow/Transforms",
        "//llvm-external-projects/iree-dialects:IREEDialectsTransforms",
        "//llvm-external-projects/iree-dialects:IREELinalgTransformDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:PDLDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)
