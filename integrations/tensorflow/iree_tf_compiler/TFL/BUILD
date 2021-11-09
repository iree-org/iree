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
    name = "TFL",
    srcs = [
        "ConvertMetadata.cpp",
        "LowerGlobalTensors.cpp",
        "Passes.cpp",
        "RetainCallOnceFuncs.cpp",
        "StripMetadata.cpp",
        "VerifyFullyConverted.cpp",
    ],
    hdrs = [
        "PassDetail.h",
        "Passes.h",
        "Passes.h.inc",
    ],
    defines = [
        "IREE_COMPILER_TENSORFLOW_ENABLED",
    ],
    deps = [
        ":PassesIncGen",
        "@iree//iree/compiler/Dialect/Flow/IR",
        "@iree//iree/compiler/Dialect/Shape/IR",
        "@iree//iree/compiler/Dialect/Shape/Transforms",
        "@iree//iree/compiler/Dialect/Util/IR",
        "@iree//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/tosa:tf_tfl_passes",
    ],
)
