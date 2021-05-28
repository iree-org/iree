# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@iree//build_tools/bazel:tblgen.bzl", "gentbl_cc_library")
load("@iree//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "td_files",
    srcs = enforce_glob(
        [
            "base.td",
            "ops.td",
        ],
        include = ["*.td"],
    ),
)

gentbl_cc_library(
    name = "ops_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "op_interface_gen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "op_interface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "op_interface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "base.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "ir",
    deps = [
        ":dialect",
        ":ops",
    ],
)

cc_library(
    name = "dialect",
    srcs = [
        "dialect.cpp",
    ],
    hdrs = [
        "dialect.h",
    ],
    deps = [
        ":op_interface_gen",
        ":ops",
        ":ops_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "ops",
    srcs = [
        "ops.cpp",
        "ops.cpp.inc",
    ],
    hdrs = [
        "ops.h",
        "ops.h.inc",
        "types.h",
    ],
    deps = [
        ":op_interface_gen",
        ":ops_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
