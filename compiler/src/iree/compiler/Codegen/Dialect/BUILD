# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_tablegen.bzl", "iree_gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "IREECodegenAttributes.td",
    "IREECodegenDialect.td",
    "LoweringConfig.td",
])

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "IREECodegenAttributes.td",
            "IREECodegenDialect.td",
            "LoweringConfig.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

iree_compiler_cc_library(
    name = "IREECodegenDialect",
    srcs = [
        "IREECodegenDialect.cpp",
        "LoweringConfig.cpp",
    ],
    hdrs = [
        "IREECodegenDialect.h",
        "LoweringConfig.h",
    ],
    textual_hdrs = [
        "IREECodegenDialect.cpp.inc",
        "IREECodegenDialect.h.inc",
        "LoweringConfig.cpp.inc",
        "LoweringConfig.h.inc",
        "LoweringConfigEnums.cpp.inc",
        "LoweringConfigEnums.h.inc",
    ],
    deps = [
        ":IREECodegenDialectGen",
        ":LoweringConfigGen",
        "//compiler/src/iree/compiler/Codegen/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

iree_gentbl_cc_library(
    name = "IREECodegenDialectGen",
    tbl_outs = [
        (
            ["--gen-dialect-decls"],
            "IREECodegenDialect.h.inc",
        ),
        (
            ["--gen-dialect-defs"],
            "IREECodegenDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IREECodegenAttributes.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "LoweringConfigGen",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "LoweringConfig.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "LoweringConfig.cpp.inc",
        ),
        (
            ["--gen-enum-decls"],
            "LoweringConfigEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "LoweringConfigEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LoweringConfig.td",
    deps = [":td_files"],
)
