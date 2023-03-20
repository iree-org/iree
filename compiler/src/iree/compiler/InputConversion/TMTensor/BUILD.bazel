# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library", "iree_gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_gentbl_cc_library(
    name = "PassesIncGen",
    tbl_outs = [
        (
            ["--gen-pass-decls"],
            "Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

iree_compiler_cc_library(
    name = "PassHeaders",
    hdrs = [
        "PassDetail.h",
        "Passes.h",
        "Passes.h.inc",
    ],
    deps = [
        ":PassesIncGen",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

iree_compiler_cc_library(
    name = "TMTensor",
    srcs = [
        "ConvertTMTensorToLinalgExt.cpp",
        "Passes.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    defines = [
        "IREE_HAVE_TORCH_INPUT",
    ],
    deps = [
        ":PassHeaders",
        ":PassesIncGen",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@torch-mlir-dialects//:TorchMLIRTMTensorDialect",
    ],
)
