# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl_cc_library(
    name = "PassesIncGen",
    tbl_outs = [
        (
            ["-gen-pass-decls"],
            "Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
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

cc_library(
    name = "TOSA",
    srcs = [
        "Passes.cpp",
        "VerifyCompilerTOSAInputLegality.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":PassHeaders",
        ":PassesIncGen",
        "//iree/compiler/Dialect/Flow/Transforms",
        "//iree/compiler/InputConversion/Common",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TosaToArith",
        "@llvm-project//mlir:TosaToLinalg",
        "@llvm-project//mlir:TosaToSCF",
        "@llvm-project//mlir:TosaToTensor",
        "@llvm-project//mlir:Transforms",
    ],
)
