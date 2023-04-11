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

iree_td_library(
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
        "//compiler/src/iree/compiler/Dialect/Util/IR:td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
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
        "//compiler/src/iree/compiler/Dialect/Stream/Builtins",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

iree_gentbl_cc_library(
    name = "StreamEnumsGen",
    tbl_outs = [
        (
            ["--gen-enum-decls"],
            "StreamEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "StreamEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StreamBase.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "StreamInterfacesGen",
    tbl_outs = [
        (
            ["--gen-attr-interface-decls"],
            "StreamAttrInterfaces.h.inc",
        ),
        (
            ["--gen-attr-interface-defs"],
            "StreamAttrInterfaces.cpp.inc",
        ),
        (
            ["--gen-op-interface-decls"],
            "StreamOpInterfaces.h.inc",
        ),
        (
            ["--gen-op-interface-defs"],
            "StreamOpInterfaces.cpp.inc",
        ),
        (
            ["--gen-type-interface-decls"],
            "StreamTypeInterfaces.h.inc",
        ),
        (
            ["--gen-type-interface-defs"],
            "StreamTypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StreamInterfaces.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "StreamOpsGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "StreamOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "StreamOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StreamOps.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "StreamTypesGen",
    tbl_outs = [
        (
            [
                "--gen-attrdef-decls",
                "--attrdefs-dialect=stream",
            ],
            "StreamAttrs.h.inc",
        ),
        (
            [
                "--gen-attrdef-defs",
                "--attrdefs-dialect=stream",
            ],
            "StreamAttrs.cpp.inc",
        ),
        (
            [
                "--gen-typedef-decls",
                "--typedefs-dialect=stream",
            ],
            "StreamTypes.h.inc",
        ),
        (
            [
                "--gen-typedef-defs",
                "--typedefs-dialect=stream",
            ],
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
            [
                "--gen-dialect-doc",
                "--dialect=stream",
            ],
            "StreamDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StreamOps.td",
    deps = [":td_files"],
)
