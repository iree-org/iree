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

filegroup(
    name = "td_files",
    srcs = enforce_glob(
        ["Ops.td"],
        include = ["*.td"],
    ),
)

cc_library(
    name = "IR",
    srcs = [
        "Ops.cc",
        "Ops.cc.inc",
    ],
    hdrs = [
        "Ops.h",
        "Ops.h.inc",
        "Types.h",
    ],
    deps = [
        ":Ops_gen",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "Dialect",
    srcs = [
        "Dialect.cc",
    ],
    hdrs = [
        "Dialect.h",
    ],
    deps = [
        ":IR",
        ":Ops_gen",
        "//iree/compiler/Dialect/HAL/Conversion",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Modules/Strings:strings_imports",
        "//iree/compiler/Dialect/Modules/Strings/Conversion",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "Ops_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "Ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Ops.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/HAL/IR:td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

iree_tablegen_doc(
    name = "DialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "StringsDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Ops.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/HAL/IR:td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)
