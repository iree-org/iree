# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")
load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_tablegen.bzl", "iree_gentbl_cc_library")

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
        "//compiler/src/iree/compiler/Dialect/HAL/IR:td_files",
        "//compiler/src/iree/compiler/Dialect/Util/IR:td_files",
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
        "//compiler/src/iree/compiler/Dialect/HAL/Conversion",
        "//compiler/src/iree/compiler/Dialect/HAL/IR",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Dialect/VM/Conversion",
        "//samples:defs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Transforms",
    ],
)

iree_gentbl_cc_library(
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
        "//samples:defs",
        "@llvm-project//mlir:IR",
    ],
)

cc_binary(
    name = "custom-opt",
    srcs = ["custom-opt-main.cc"],
    deps = [
        ":init_dialect",
        "//compiler/src/iree/compiler/Tools:init_passes_and_dialects",
        "//compiler/src/iree/compiler/Tools:init_targets",
        "//samples:defs",
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
        "//compiler/src/iree/compiler/Dialect/VM/Target:init_targets",
        "//compiler/src/iree/compiler/Dialect/VM/Target/Bytecode",
        "//compiler/src/iree/compiler/Tools:init_compiler_modules",
        "//compiler/src/iree/compiler/Tools:init_iree_passes_and_dialects",
        "//compiler/src/iree/compiler/Tools:init_mlir_passes_and_dialects",
        "//compiler/src/iree/compiler/Tools:init_targets",
        "//compiler/src/iree/compiler/Tools:init_translations",
        "//compiler/src/iree/compiler/Tools:init_xla_dialects",
        "//compiler/src/iree/compiler/Translation:IREEVM",
        "//samples:defs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)
