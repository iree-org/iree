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

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "FlowBase.td",
            "FlowInterfaces.td",
            "FlowOps.td",
        ],
        include = ["*.td"],
        exclude = ["PartitionableLoopsInterface.td"],
    ),
    deps = [
        "//iree/compiler/Dialect/Util/IR:td_files",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

cc_library(
    name = "IR",
    srcs = [
        "FlowDialect.cpp",
        "FlowEnums.cpp.inc",
        "FlowOpFolders.cpp",
        "FlowOpInterfaces.cpp.inc",
        "FlowOps.cpp",
        "FlowOps.cpp.inc",
        "FlowTypeInterfaces.cpp.inc",
        "FlowTypes.cpp",
        "FlowTypes.cpp.inc",
    ],
    hdrs = [
        "FlowDialect.h",
        "FlowEnums.h.inc",
        "FlowOpInterfaces.h.inc",
        "FlowOps.h",
        "FlowOps.h.inc",
        "FlowTypeInterfaces.h.inc",
        "FlowTypes.h",
        "FlowTypes.h.inc",
    ],
    deps = [
        ":FlowEnumsGen",
        ":FlowInterfacesGen",
        ":FlowOpsGen",
        ":FlowTypesGen",
        "//iree/compiler/Dialect/Util/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticUtils",
        "@llvm-project//mlir:ControlFlowInterfaces",
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
    name = "FlowEnumsGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "FlowEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "FlowEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowBase.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "FlowInterfacesGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "FlowOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "FlowOpInterfaces.cpp.inc",
        ),
        (
            ["-gen-type-interface-decls"],
            "FlowTypeInterfaces.h.inc",
        ),
        (
            ["-gen-type-interface-defs"],
            "FlowTypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowInterfaces.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "FlowOpsGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "FlowOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "FlowOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowOps.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "FlowTypesGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "FlowAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "FlowAttrs.cpp.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "FlowTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "FlowTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowBase.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "FlowDialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "FlowDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowOps.td",
    deps = [":td_files"],
)

cc_library(
    name = "PartitionableLoopsInterface",
    srcs = [
        "PartitionableLoopsInterface.cpp",
    ],
    hdrs = [
        "PartitionableLoopsInterface.h",
    ],
    textual_hdrs = [
        "PartitionableLoopsInterface.cpp.inc",
        "PartitionableLoopsInterface.h.inc",
    ],
    deps = [
        ":PartitionableLoopsInterfaceGen",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:LinalgInterfaces",
        "@llvm-project//mlir:LinalgOps",
    ],
)

gentbl_cc_library(
    name = "PartitionableLoopsInterfaceGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "PartitionableLoopsInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "PartitionableLoopsInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PartitionableLoopsInterface.td",
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
)
