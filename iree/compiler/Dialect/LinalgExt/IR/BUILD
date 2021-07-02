# Copyright 2021 The IREE Authors
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

exports_files(["LinalgExtBase.td"])

filegroup(
    name = "td_files",
    srcs = enforce_glob(
        [
            "LinalgExtBase.td",
            "LinalgExtOps.td",
            "LinalgExtInterfaces.td",
        ],
        include = ["*.td"],
    ),
)

cc_library(
    name = "IR",
    srcs = [
        "LinalgExtDialect.cpp",
        "LinalgExtInterfaces.cpp",
        "LinalgExtInterfaces.cpp.inc",
        "LinalgExtOps.cpp",
        "LinalgExtOps.cpp.inc",
    ],
    hdrs = [
        "LinalgExtDialect.h",
        "LinalgExtInterfaces.h",
        "LinalgExtInterfaces.h.inc",
        "LinalgExtOps.h",
        "LinalgExtOps.h.inc",
    ],
    deps = [
        ":LinalgExtInterfacesGen",
        ":LinalgExtOpsGen",
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
    name = "LinalgExtOpsGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "LinalgExtOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "LinalgExtOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgExtOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "LinalgExtInterfacesGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "LinalgExtInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "LinalgExtInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgExtInterfaces.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

iree_tablegen_doc(
    name = "LinalgExtDialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "LinalgExtDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgExtOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
    ],
)
