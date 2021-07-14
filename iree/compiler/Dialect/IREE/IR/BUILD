# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:iree_tablegen_doc.bzl", "iree_tablegen_doc")
load("//build_tools/bazel:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["IREEBase.td"])

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "IREEBase.td",
            "IREEInterfaces.td",
            "IREEOps.td",
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
        "IREEDialect.cpp",
        "IREEOpInterfaces.cpp.inc",
        "IREEOps.cpp",
        "IREEOps.cpp.inc",
        "IREETypes.cpp",
    ],
    hdrs = [
        "IREEDialect.h",
        "IREEOpInterfaces.h.inc",
        "IREEOps.h",
        "IREEOps.h.inc",
        "IREETraits.h",
        "IREETypes.h",
    ],
    deps = [
        ":IREEInterfacesGen",
        ":IREEOpsGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "IREEInterfacesGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "IREEOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "IREEOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IREEInterfaces.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "IREEOpsGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "IREEOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "IREEOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IREEOps.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "IREEDialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "IREEDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IREEOps.td",
    deps = [":td_files"],
)
