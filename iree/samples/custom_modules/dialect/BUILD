# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")
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
        ["custom_ops.td"],
        include = ["*.td"],
    ),
    deps = [
        "//iree/compiler/Dialect/HAL/IR:td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_library(
    name = "dialect",
    srcs = [
        "conversion_patterns.cc",
        "custom_dialect.cc",
        "custom_ops.cc.inc",
    ],
    hdrs = [
        "conversion_patterns.h",
        "custom_dialect.h",
        "custom_ops.h.inc",
    ],
    deps = [
        ":custom_imports",
        ":custom_ops_gen",
        "//iree/compiler/Dialect/HAL/Conversion",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "custom_ops_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "custom_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "custom_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "custom_ops.td",
    deps = [":td_files"],
)

c_embed_data(
    name = "custom_imports",
    srcs = ["custom.imports.mlir"],
    c_file_output = "custom.imports.c",
    flatten = True,
    h_file_output = "custom.imports.h",
    identifier = "iree_custom_imports",
)

cc_library(
    name = "init_dialect",
    hdrs = ["init_dialect.h"],
    deps = [
        ":dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_binary(
    name = "custom-opt",
    srcs = ["custom-opt-main.cc"],
    deps = [
        ":init_dialect",
        "//iree/tools:init_passes_and_dialects",
        "//iree/tools:init_targets",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Support",
    ],
)

cc_binary(
    name = "custom-translate",
    srcs = ["custom-translate-main.cc"],
    deps = [
        ":init_dialect",
        "//iree/compiler/Dialect/VM/Target:init_targets",
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/compiler/Translation:IREEVM",
        "//iree/tools:init_compiler_modules",
        "//iree/tools:init_iree_passes_and_dialects",
        "//iree/tools:init_mlir_passes_and_dialects",
        "//iree/tools:init_targets",
        "//iree/tools:init_translations",
        "//iree/tools:init_xla_dialects",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)
