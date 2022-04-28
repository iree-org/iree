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

exports_files(["VMLXOps.td"])

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "VMVXBase.td",
            "VMVXOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "//iree/compiler/Dialect/Util/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_library(
    name = "IR",
    srcs = [
        "VMVXEnums.cpp.inc",
        "VMVXOpInterface.cpp.inc",
        "VMVXOps.cpp",
        "VMVXTypes.cpp",
    ],
    hdrs = [
        "VMVXEnums.h.inc",
        "VMVXOpInterface.h.inc",
        "VMVXOps.h",
        "VMVXOps.h.inc",
        "VMVXTypes.h",
    ],
    textual_hdrs = [
        "VMVXOps.cpp.inc",
    ],
    deps = [
        ":VMVXEnumsGen",
        ":VMVXOpInterfaceGen",
        ":VMVXOpsGen",
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Translation",
    ],
)

cc_library(
    name = "VMVXDialect",
    srcs = ["VMVXDialect.cpp"],
    hdrs = ["VMVXDialect.h"],
    deps = [
        ":IR",
        "//iree/compiler/Dialect/Modules/VMVX:vmvx_imports",
        "//iree/compiler/Dialect/Modules/VMVX/Conversion/VMVXToVM",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "VMVXEnumsGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "VMVXEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "VMVXEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXBase.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "VMVXOpsGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "VMVXOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "VMVXOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXOps.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "VMVXOpInterfaceGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "VMVXOpInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "VMVXOpInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXBase.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "VMVXDialecDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "VMVXDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXOps.td",
    deps = [":td_files"],
)
