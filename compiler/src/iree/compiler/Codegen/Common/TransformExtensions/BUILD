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
            "CommonExtensionsOps.td",
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
    name = "CommonExtensionsOpGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "CommonExtensionsOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "CommonExtensionsOps.cpp.inc",
        ),
        (
            ["--gen-enum-decls"],
            "CommonExtensionsAttrs.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "CommonExtensionsAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CommonExtensionsOps.td",
    deps = [":td_files"],
)

iree_compiler_cc_library(
    name = "CommonExtensions",
    srcs = [
        "CommonExtensions.cpp",
        "CommonExtensionsAttrs.cpp.inc",
        "CommonExtensionsOps.cpp.inc",
    ],
    hdrs = [
        "CommonExtensions.h",
        "CommonExtensionsAttrs.h.inc",
        "CommonExtensionsOps.h.inc",
    ],
    deps = [
        ":CommonExtensionsOpGen",
        "//compiler/src/iree/compiler/Codegen:PassHeaders",
        "//compiler/src/iree/compiler/Codegen/Interfaces:BufferizationInterfaces",
        "//llvm-external-projects/iree-dialects:IREEDialectsTransforms",
        "//llvm-external-projects/iree-dialects:IREELinalgTransformDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformDialect",
    ],
)
