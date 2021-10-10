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

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "StreamBase.td",
            "StreamInterfaces.td",
            "StreamOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "//iree/compiler/Dialect/Shape/IR:td_files",
        "//iree/compiler/Dialect/Util/IR:td_files",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:SubElementInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

cc_library(
    name = "IR",
    srcs = [
        "StreamDialect.cpp",
        "StreamEnums.cpp.inc",
        "StreamOpFolders.cpp",
        "StreamOpInterfaces.cpp.inc",
        "StreamOps.cpp",
        "StreamOps.cpp.inc",
        "StreamTypeInterfaces.cpp.inc",
        "StreamTypes.cpp",
        "StreamTypes.cpp.inc",
    ],
    hdrs = [
        "StreamDialect.h",
        "StreamEnums.h.inc",
        "StreamOpInterfaces.h.inc",
        "StreamOps.h",
        "StreamOps.h.inc",
        "StreamTraits.h",
        "StreamTypeInterfaces.h.inc",
        "StreamTypes.h",
        "StreamTypes.h.inc",
    ],
    deps = [
        ":StreamEnumsGen",
        ":StreamInterfacesGen",
        ":StreamOpsGen",
        ":StreamTypesGen",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Util/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "StreamEnumsGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "StreamEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "StreamEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StreamBase.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "StreamInterfacesGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "StreamOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "StreamOpInterfaces.cpp.inc",
        ),
        (
            ["-gen-type-interface-decls"],
            "StreamTypeInterfaces.h.inc",
        ),
        (
            ["-gen-type-interface-defs"],
            "StreamTypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StreamInterfaces.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "StreamOpsGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "StreamOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "StreamOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StreamOps.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "StreamTypesGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "StreamAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "StreamAttrs.cpp.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "StreamTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "StreamTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StreamBase.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "StreamDialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "StreamDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StreamOps.td",
    deps = [":td_files"],
)
