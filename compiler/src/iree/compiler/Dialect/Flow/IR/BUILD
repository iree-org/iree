# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_tablegen.bzl", "iree_gentbl_cc_library", "iree_tablegen_doc")

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
        "//compiler/src/iree/compiler/Dialect/Util/IR:td_files",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

iree_compiler_cc_library(
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
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ArithmeticUtils",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

iree_gentbl_cc_library(
    name = "FlowEnumsGen",
    tbl_outs = [
        (
            ["--gen-enum-decls"],
            "FlowEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "FlowEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowBase.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "FlowInterfacesGen",
    tbl_outs = [
        (
            ["--gen-op-interface-decls"],
            "FlowOpInterfaces.h.inc",
        ),
        (
            ["--gen-op-interface-defs"],
            "FlowOpInterfaces.cpp.inc",
        ),
        (
            ["--gen-type-interface-decls"],
            "FlowTypeInterfaces.h.inc",
        ),
        (
            ["--gen-type-interface-defs"],
            "FlowTypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowInterfaces.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "FlowOpsGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "FlowOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "FlowOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowOps.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "FlowTypesGen",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "FlowAttrs.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "FlowAttrs.cpp.inc",
        ),
        (
            ["--gen-typedef-decls"],
            "FlowTypes.h.inc",
        ),
        (
            ["--gen-typedef-defs"],
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
            [
                "--gen-dialect-doc",
                "-dialect=flow",
            ],
            "FlowDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowOps.td",
    deps = [":td_files"],
)

iree_compiler_cc_library(
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgStructuredOpsIncGen",
        "@llvm-project//mlir:TensorDialect",
    ],
)

iree_gentbl_cc_library(
    name = "PartitionableLoopsInterfaceGen",
    tbl_outs = [
        (
            ["--gen-op-interface-decls"],
            "PartitionableLoopsInterface.h.inc",
        ),
        (
            ["--gen-op-interface-defs"],
            "PartitionableLoopsInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PartitionableLoopsInterface.td",
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
)
