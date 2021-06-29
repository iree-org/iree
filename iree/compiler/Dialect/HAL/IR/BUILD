# Copyright 2019 The IREE Authors
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

exports_files([
    "HALBase.td",
    "HALDialect.td",
])

filegroup(
    name = "td_files",
    srcs = enforce_glob(
        [
            "HALBase.td",
            "HALDialect.td",
            "HALInterfaces.td",
            "HALOps.td",
            "LoweringConfig.td",
        ],
        include = ["*.td"],
    ),
)

cc_library(
    name = "IR",
    srcs = [
        "HALOpFolders.cpp",
        "HALOps.cpp",
        "HALTypes.cpp",
        "LoweringConfig.cpp",
    ],
    hdrs = [
        "HALDialect.h",
        "HALOps.h",
        "HALTraits.h",
        "HALTypes.h",
        "LoweringConfig.h",
    ],
    textual_hdrs = [
        "HALEnums.cpp.inc",
        "HALEnums.h.inc",
        "HALOpInterfaces.cpp.inc",
        "HALOpInterfaces.h.inc",
        "HALOps.cpp.inc",
        "HALOps.h.inc",
        "HALStructs.cpp.inc",
        "HALStructs.h.inc",
        "HALTypeInterfaces.cpp.inc",
        "HALTypeInterfaces.h.inc",
        "LoweringConfig.h.inc",
        "LoweringConfig.cpp.inc",
        "LoweringConfigEnums.h.inc",
        "LoweringConfigEnums.cpp.inc",
    ],
    deps = [
        ":HALInterfacesGen",
        ":HALOpsGen",
        ":HALStructsGen",
        ":HALTypesGen",
        ":LoweringConfigEnumGen",
        ":LoweringConfigGen",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "HALDialect",
    srcs = ["HALDialect.cpp"],
    hdrs = ["HALDialect.h"],
    deps = [
        ":IR",
        "//iree/compiler/Dialect/HAL:hal_imports",
        "//iree/compiler/Dialect/HAL/Conversion/HALToHAL",
        "//iree/compiler/Dialect/HAL/Conversion/HALToVM",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "HALInterfacesGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "HALOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "HALOpInterfaces.cpp.inc",
        ),
        (
            ["-gen-type-interface-decls"],
            "HALTypeInterfaces.h.inc",
        ),
        (
            ["-gen-type-interface-defs"],
            "HALTypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HALInterfaces.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "HALOpsGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "HALOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "HALOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HALOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/Shape/IR:td_files",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "HALStructsGen",
    tbl_outs = [
        (
            ["-gen-iree-struct-attr-decls"],
            "HALStructs.h.inc",
        ),
        (
            ["-gen-iree-struct-attr-defs"],
            "HALStructs.cpp.inc",
        ),
    ],
    tblgen = "//iree/tools:iree-tblgen",
    td_file = "HALBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "HALTypesGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "HALEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "HALEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HALBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

iree_tablegen_doc(
    name = "HALDialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "HALDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HALOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/Shape/IR:td_files",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "LoweringConfigGen",
    tbl_outs = [
        (
            ["-gen-struct-attr-decls"],
            "LoweringConfig.h.inc",
        ),
        (
            ["-gen-struct-attr-defs"],
            "LoweringConfig.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LoweringConfig.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "LoweringConfigEnumGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "LoweringConfigEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "LoweringConfigEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LoweringConfig.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)
