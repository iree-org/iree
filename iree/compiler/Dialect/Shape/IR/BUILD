# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:iree_tablegen_doc.bzl", "iree_tablegen_doc")
load("//build_tools/bazel:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["ShapeBase.td"])

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "ShapeBase.td",
            "ShapeInterfaces.td",
            "ShapeOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "//iree/compiler/Dialect/IREE/IR:td_files",
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
        "Builders.cpp",
        "Folders.cpp",
        "ShapeDialect.cpp",
        "ShapeInterfaces.cpp.inc",
        "ShapeOps.cpp",
        "ShapeOps.cpp.inc",
        "ShapeTypes.cpp",
    ],
    hdrs = [
        "Builders.h",
        "ShapeDialect.h",
        "ShapeInterfaces.h.inc",
        "ShapeOps.h",
        "ShapeOps.h.inc",
        "ShapeTypes.h",
    ],
    deps = [
        ":ShapeInterfacesGen",
        ":ShapeOpsGen",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

gentbl_cc_library(
    name = "ShapeInterfacesGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "ShapeInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "ShapeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ShapeInterfaces.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "ShapeOpsGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ShapeOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ShapeOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ShapeOps.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "ShapeDialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "ShapeDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ShapeOps.td",
    deps = [":td_files"],
)
