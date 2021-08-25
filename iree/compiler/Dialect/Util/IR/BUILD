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
            "UtilAttrs.td",
            "UtilBase.td",
            "UtilInterfaces.td",
            "UtilOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:SubElementInterfacesTdFiles",
    ],
)

cc_library(
    name = "IR",
    srcs = [
        "ClosureOpUtils.cpp",
        "UtilAttrs.cpp",
        "UtilDialect.cpp",
        "UtilOpFolders.cpp",
        "UtilOps.cpp",
        "UtilOps.cpp.inc",
        "UtilTypes.cpp",
    ],
    hdrs = [
        "ClosureOpUtils.h",
        "UtilDialect.h",
        "UtilOps.h",
        "UtilOps.h.inc",
        "UtilTraits.h",
        "UtilTypes.h",
    ],
    textual_hdrs = [
        "UtilAttrs.cpp.inc",
        "UtilAttrs.h.inc",
        "UtilAttrInterfaces.cpp.inc",
        "UtilAttrInterfaces.h.inc",
        "UtilOpInterfaces.cpp.inc",
        "UtilOpInterfaces.h.inc",
        "UtilTypeInterfaces.cpp.inc",
        "UtilTypeInterfaces.h.inc",
    ],
    deps = [
        ":UtilAttrsGen",
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
    name = "UtilAttrsGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "UtilAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "UtilAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "UtilAttrs.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "UtilInterfacesGen",
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "UtilAttrInterfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "UtilAttrInterfaces.cpp.inc",
        ),
        (
            ["-gen-op-interface-decls"],
            "UtilOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "UtilOpInterfaces.cpp.inc",
        ),
        (
            ["-gen-type-interface-decls"],
            "UtilTypeInterfaces.h.inc",
        ),
        (
            ["-gen-type-interface-defs"],
            "UtilTypeInterfaces.cpp.inc",
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
