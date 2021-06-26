# Copyright 2021 The LinalgPlus Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:iree_tablegen_doc.bzl", "iree_tablegen_doc")
load("//build_tools/bazel:tblgen.bzl", "gentbl_cc_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LinalgPlusBase.td"])

filegroup(
    name = "td_files",
    srcs = enforce_glob(
        [
            "LinalgPlusBase.td",
            "LinalgPlusOps.td",
            "LinalgPlusInterfaces.td",
        ],
        include = ["*.td"],
    ),
)

cc_library(
    name = "IR",
    srcs = [
        "LinalgPlusDialect.cpp",
        "LinalgPlusInterfaces.cpp",
        "LinalgPlusInterfaces.cpp.inc",
        "LinalgPlusOps.cpp",
        "LinalgPlusOps.cpp.inc",
    ],
    hdrs = [
        "LinalgPlusDialect.h",
        "LinalgPlusInterfaces.h",
        "LinalgPlusInterfaces.h.inc",
        "LinalgPlusOps.h",
        "LinalgPlusOps.h.inc",
    ],
    deps = [
        ":LinalgPlusInterfacesGen",
        ":LinalgPlusOpsGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "LinalgPlusOpsGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "LinalgPlusOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "LinalgPlusOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgPlusOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "LinalgPlusInterfacesGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "LinalgPlusInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "LinalgPlusInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgPlusInterfaces.td",
    td_includes = ["third_party/iree"],
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

iree_tablegen_doc(
    name = "LinalgPlusDialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "LinalgPlusDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgPlusOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
    ],
)
