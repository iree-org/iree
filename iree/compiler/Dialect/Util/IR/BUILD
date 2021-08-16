# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:iree_tablegen_doc.bzl", "iree_tablegen_doc")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["UtilBase.td"])

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "UtilBase.td",
            "UtilInterfaces.td",
            "UtilOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
)

cc_library(
    name = "IR",
    srcs = [
        "UtilDialect.cpp",
        "UtilOpFolders.cpp",
        "UtilOpInterfaces.cpp.inc",
        "UtilOps.cpp",
        "UtilOps.cpp.inc",
        "UtilTypes.cpp",
    ],
    hdrs = [
        "UtilDialect.h",
        "UtilOpInterfaces.h.inc",
        "UtilOps.h",
        "UtilOps.h.inc",
        "UtilTraits.h",
        "UtilTypes.h",
    ],
    deps = [
        ":UtilInterfacesGen",
        ":UtilOpsGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "UtilInterfacesGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "UtilOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "UtilOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "UtilInterfaces.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "UtilOpsGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "UtilOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "UtilOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "UtilOps.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "UtilUtilDialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "UtilUtilDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "UtilOps.td",
    deps = [":td_files"],
)
