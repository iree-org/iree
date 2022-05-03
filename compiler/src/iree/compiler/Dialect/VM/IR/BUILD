# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:iree_tablegen.bzl", "iree_gentbl_cc_library", "iree_tablegen_doc")
load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//iree:build_defs.oss.bzl", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["VMOps.td"])

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "VMBase.td",
            "VMOpcodesCore.td",
            "VMOpcodesF32.td",
            "VMOpcodesF64.td",
            "VMOpcodesI64.td",
            "VMOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "//compiler/src/iree/compiler/Dialect/Util/IR:td_files",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
)

iree_compiler_cc_library(
    name = "IR",
    srcs = [
        "VMDialect.cpp",
        "VMEnums.cpp.inc",
        "VMOpEncoder.cpp.inc",
        "VMOpFolders.cpp",
        "VMOpInterface.cpp.inc",
        "VMOps.cpp",
        "VMOps.cpp.inc",
        "VMStructs.cpp.inc",
        "VMTypes.cpp",
    ],
    hdrs = [
        "VMDialect.h",
        "VMEnums.h.inc",
        "VMFuncEncoder.h",
        "VMOpInterface.h.inc",
        "VMOps.h",
        "VMOps.h.inc",
        "VMStructs.h.inc",
        "VMTraits.h",
        "VMTypes.h",
    ],
    deps = [
        ":VMEnumsGen",
        ":VMOpEncoderGen",
        ":VMOpInterfaceGen",
        ":VMOpsGen",
        ":VMStructsGen",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

iree_gentbl_cc_library(
    name = "VMEnumsGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "VMEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
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
            ["-gen-op-decls"],
            "VMOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
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
            ["-gen-iree-vm-op-encoder-defs"],
            "VMOpEncoder.cpp.inc",
        ),
    ],
    tblgen = "//iree/tools:iree-tblgen",
    td_file = "VMOps.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "VMOpInterfaceGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "VMOpInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "VMOpInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMBase.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "VMStructsGen",
    tbl_outs = [
        (
            ["-gen-iree-struct-attr-decls"],
            "VMStructs.h.inc",
        ),
        (
            ["-gen-iree-struct-attr-defs"],
            "VMStructs.cpp.inc",
        ),
    ],
    tblgen = "//iree/tools:iree-tblgen",
    td_file = "VMBase.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "VMDialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "VMDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMOps.td",
    deps = [":td_files"],
)
