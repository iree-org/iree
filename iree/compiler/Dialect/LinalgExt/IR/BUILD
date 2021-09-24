# Copyright 2021 The IREE Authors
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

exports_files(["LinalgExtBase.td"])

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "LinalgExtBase.td",
            "LinalgExtOps.td",
            "LinalgExtInterfaces.td",
            "TiledOpInterface.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
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
        ":TiledOpInterface",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
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
        (
            [
                "-gen-dialect-decls",
            ],
            "LinalgExtOpsDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
            ],
            "LinalgExtOpsDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgExtOps.td",
    deps = [":td_files"],
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
    deps = [":td_files"],
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
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "TiledOpInterfaceGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "TiledOpInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "TiledOpInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TiledOpInterface.td",
    deps = [":td_files"],
)

cc_library(
    name = "TiledOpInterface",
    srcs = [
        "TiledOpInterface.cpp",
        "TiledOpInterface.cpp.inc",
    ],
    hdrs = [
        "TiledOpInterface.h",
        "TiledOpInterface.h.inc",
    ],
    deps = [
        ":TiledOpInterfaceGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)
