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
    "MicroKernelOps.td",
])

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "IREECodegenAttributes.td",
            "IREECodegenDialect.td",
            "LoweringConfig.td",
            "MicroKernelOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "//compiler/src/iree/compiler/Codegen/Interfaces:td_files",
        "@llvm-project//mlir:DestinationStyleOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

iree_compiler_cc_library(
    name = "IREECodegenDialect",
    srcs = [
        "IREECodegenDialect.cpp",
        "LoweringConfig.cpp",
        "MicroKernelOps.cpp",
    ],
    hdrs = [
        "IREECodegenDialect.h",
        "LoweringConfig.h",
        "MicroKernelOps.h",
    ],
    textual_hdrs = [
        "IREECodegenDialect.cpp.inc",
        "IREECodegenDialect.h.inc",
        "LoweringConfig.cpp.inc",
        "LoweringConfig.h.inc",
        "LoweringConfigEnums.cpp.inc",
        "LoweringConfigEnums.h.inc",
        "MicroKernelOps.cpp.inc",
        "MicroKernelOps.h.inc",
    ],
    deps = [
        ":IREECodegenDialectGen",
        ":LoweringConfigGen",
        ":MicroKernelOpsGen",
        "//compiler/src/iree/compiler/Codegen/Interfaces:MicroKernelOpInterface",
        "//compiler/src/iree/compiler/Codegen/Utils",
        "//compiler/src/iree/compiler/Dialect/HAL/IR",
        "//runtime/src/iree/builtins/ukernel:exported_bits",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
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

iree_gentbl_cc_library(
    name = "MicroKernelOpsGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "MicroKernelOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "MicroKernelOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MicroKernelOps.td",
    deps = [
        ":td_files",
        "//compiler/src/iree/compiler/Codegen/Interfaces:td_files",
    ],
)
