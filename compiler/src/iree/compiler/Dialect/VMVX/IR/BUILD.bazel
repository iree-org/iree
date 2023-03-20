# Copyright 2021 The IREE Authors
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

exports_files(["VMLXOps.td"])

iree_td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "VMVXBase.td",
            "VMVXInterfaces.td",
            "VMVXOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "//compiler/src/iree/compiler/Dialect/Util/IR:td_files",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

iree_compiler_cc_library(
    name = "IR",
    srcs = [
        "VMVXEnums.cpp.inc",
        "VMVXOps.cpp",
        "VMVXTypes.cpp",
    ],
    hdrs = [
        "VMVXEnums.h.inc",
        "VMVXOps.h",
        "VMVXTypes.h",
    ],
    textual_hdrs = [
        "VMVXOpInterfaces.cpp.inc",
        "VMVXOpInterfaces.h.inc",
        "VMVXOps.cpp.inc",
        "VMVXOps.h.inc",
    ],
    deps = [
        ":VMVXEnumsGen",
        ":VMVXInterfacesGen",
        ":VMVXOpsGen",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:TranslateLib",
    ],
)

iree_compiler_cc_library(
    name = "VMVXDialect",
    srcs = ["VMVXDialect.cpp"],
    hdrs = ["VMVXDialect.h"],
    deps = [
        ":IR",
        "//compiler/src/iree/compiler/Dialect/VM/Conversion",
        "//compiler/src/iree/compiler/Dialect/VMVX:vmvx_imports",
        "//compiler/src/iree/compiler/Dialect/VMVX/Conversion/VMVXToVM",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

iree_gentbl_cc_library(
    name = "VMVXEnumsGen",
    tbl_outs = [
        (
            ["--gen-enum-decls"],
            "VMVXEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "VMVXEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXBase.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "VMVXOpsGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "VMVXOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "VMVXOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXOps.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "VMVXInterfacesGen",
    tbl_outs = [
        (
            ["--gen-op-interface-decls"],
            "VMVXOpInterfaces.h.inc",
        ),
        (
            ["--gen-op-interface-defs"],
            "VMVXOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXInterfaces.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "VMVXDialectDocGen",
    tbl_outs = [
        (
            [
                "--dialect=vmvx",
                "--gen-dialect-doc",
            ],
            "VMVXDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXOps.td",
    deps = [":td_files"],
)
