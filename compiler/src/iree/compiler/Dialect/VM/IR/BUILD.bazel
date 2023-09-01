# Copyright 2019 The IREE Authors
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

exports_files(["VMOps.td"])

iree_td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "VMBase.td",
            "VMOpcodesCore.td",
            "VMOpcodesF32.td",
            "VMOpcodesF64.td",
            "VMOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "//compiler/src/iree/compiler/Dialect/Util/IR:td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

iree_compiler_cc_library(
    name = "IR",
    srcs = [
        "VMDialect.cpp",
        "VMOpFolders.cpp",
        "VMOps.cpp",
        "VMTypes.cpp",
    ],
    hdrs = [
        "VMDialect.h",
        "VMFuncEncoder.h",
        "VMOps.h",
        "VMTraits.h",
        "VMTypes.h",
    ],
    textual_hdrs = [
        "VMAttrs.cpp.inc",
        "VMAttrs.h.inc",
        "VMEnums.cpp.inc",
        "VMEnums.h.inc",
        "VMOpEncoder.cpp.inc",
        "VMOpInterfaces.cpp.inc",
        "VMOpInterfaces.h.inc",
        "VMOps.cpp.inc",
        "VMOps.h.inc",
    ],
    deps = [
        ":VMAttrsGen",
        ":VMEnumsGen",
        ":VMOpEncoderGen",
        ":VMOpInterfacesGen",
        ":VMOpsGen",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

iree_gentbl_cc_library(
    name = "VMAttrsGen",
    tbl_outs = [
        (
            [
                "--gen-attrdef-decls",
                "--attrdefs-dialect=vm",
            ],
            "VMAttrs.h.inc",
        ),
        (
            [
                "--gen-attrdef-defs",
                "--attrdefs-dialect=vm",
            ],
            "VMAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMBase.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "VMEnumsGen",
    tbl_outs = [
        (
            ["--gen-enum-decls"],
            "VMEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "VMEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMBase.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "VMOpsGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "VMOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "VMOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMOps.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "VMOpEncoderGen",
    tbl_outs = [
        (
            ["--gen-iree-vm-op-encoder-defs"],
            "VMOpEncoder.cpp.inc",
        ),
    ],
    tblgen = "//tools:iree-tblgen",
    td_file = "VMOps.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "VMOpInterfacesGen",
    tbl_outs = [
        (
            ["--gen-op-interface-decls"],
            "VMOpInterfaces.h.inc",
        ),
        (
            ["--gen-op-interface-defs"],
            "VMOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMBase.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "VMDialectDocGen",
    tbl_outs = [
        (
            [
                "--gen-dialect-doc",
                "--dialect=vm",
            ],
            "VMDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMOps.td",
    deps = [":td_files"],
)
