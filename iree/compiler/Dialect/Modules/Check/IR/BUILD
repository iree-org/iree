# Copyright 2020 The IREE Authors
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

td_library(
    name = "td_files",
    srcs = enforce_glob(
        ["CheckOps.td"],
        include = ["*.td"],
    ),
    deps = [
        "//iree/compiler/Dialect/HAL/IR:td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "IR",
    srcs = [
        "CheckOps.cpp",
        "CheckOps.cpp.inc",
    ],
    hdrs = [
        "CheckOps.h",
        "CheckOps.h.inc",
    ],
    deps = [
        ":check_ops_gen",
        "//iree/compiler/Dialect/HAL/IR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "CheckDialect",
    srcs = [
        "CheckDialect.cpp",
    ],
    hdrs = [
        "CheckDialect.h",
    ],
    deps = [
        ":IR",
        ":check_ops_gen",
        "//iree/compiler/Dialect/HAL/Conversion",
        "//iree/compiler/Dialect/Modules/Check:check_imports",
        "//iree/compiler/Dialect/Modules/Check/Conversion",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "check_ops_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "CheckOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "CheckOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CheckOps.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "CheckDialectDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "CheckDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CheckOps.td",
    deps = [":td_files"],
)
