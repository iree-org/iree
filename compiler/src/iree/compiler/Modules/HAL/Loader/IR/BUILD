# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library", "iree_gentbl_cc_library", "iree_tablegen_doc", "iree_td_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["HALLoaderOps.td"])

iree_td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "HALLoaderBase.td",
            "HALLoaderOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "//compiler/src/iree/compiler/Dialect/HAL/IR:td_files",
        "//compiler/src/iree/compiler/Dialect/Util/IR:td_files",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

iree_compiler_cc_library(
    name = "IR",
    srcs = [
        "HALLoaderOps.cpp",
    ],
    hdrs = [
        "HALLoaderOps.h",
        "HALLoaderOps.h.inc",
    ],
    textual_hdrs = [
        "HALLoaderOps.cpp.inc",
    ],
    deps = [
        ":HALLoaderOpsGen",
        "//compiler/src/iree/compiler/Dialect/HAL/IR",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:TranslateLib",
    ],
)

iree_compiler_cc_library(
    name = "HALLoaderDialect",
    srcs = ["HALLoaderDialect.cpp"],
    hdrs = ["HALLoaderDialect.h"],
    deps = [
        ":IR",
        "//compiler/src/iree/compiler/Dialect/VM/Conversion",
        "//compiler/src/iree/compiler/Modules/HAL/Loader:hal_loader_imports",
        "//compiler/src/iree/compiler/Modules/HAL/Loader/Conversion/HALLoaderToVM",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

iree_gentbl_cc_library(
    name = "HALLoaderOpsGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "HALLoaderOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "HALLoaderOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HALLoaderOps.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "HALLoaderDialecDocGen",
    tbl_outs = [
        (
            [
                "--dialect=hal_loader",
                "--gen-dialect-doc",
            ],
            "HALLoaderDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HALLoaderOps.td",
    deps = [":td_files"],
)
