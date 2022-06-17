# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_tablegen.bzl", "iree_gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "TransformDialectCommonExtensionsOps.td",
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
    name = "TransformDialectCommonExtensionsOpGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "TransformDialectCommonExtensionsOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "TransformDialectCommonExtensionsOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TransformDialectCommonExtensionsOps.td",
    deps = [":td_files"],
)

iree_compiler_cc_library(
    name = "TransformDialectCommonExtensions",
    srcs = [
        "TransformDialectCommonExtensions.cpp",
        "TransformDialectCommonExtensionsOps.cpp.inc",
    ],
    hdrs = [
        "TransformDialectCommonExtensions.h",
        "TransformDialectCommonExtensionsOps.h.inc",
    ],
    deps = [
        ":TransformDialectCommonExtensionsOpGen",
        "//compiler/src/iree/compiler/Codegen",
        "//compiler/src/iree/compiler/Codegen:PassHeaders",
        "//compiler/src/iree/compiler/Codegen/Common",
        "//compiler/src/iree/compiler/Codegen/Interfaces:BufferizationInterfaces",
        "//compiler/src/iree/compiler/Codegen/Transforms",
        "//compiler/src/iree/compiler/Codegen/Utils",
        "//llvm-external-projects/iree-dialects:IREEDialectsTransforms",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "//llvm-external-projects/iree-dialects:IREELinalgExtTransforms",
        "//llvm-external-projects/iree-dialects:IREELinalgTransformDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:PDLDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)
