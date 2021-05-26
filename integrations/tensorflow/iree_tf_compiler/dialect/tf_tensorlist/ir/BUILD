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

exports_files(["tf_tensorlist_base.td"])

filegroup(
    name = "td_files",
    srcs = enforce_glob(
        [
            "tf_tensorlist_base.td",
            "tf_tensorlist_ops.td",
        ],
        include = ["*.td"],
    ),
)

cc_library(
    name = "ir",
    srcs = [
        "tf_tensorlist_ops.cc",
        "tf_tensorlist_types.cc",
    ],
    hdrs = [
        "tf_tensorlist_ops.h",
        "tf_tensorlist_ops.h.inc",
        "tf_tensorlist_types.h",
    ],
    textual_hdrs = [
        "tf_tensorlist_ops.cc.inc",
    ],
    deps = [
        ":tf_tensorlist_ops_gen",
        "@iree//iree/compiler/Dialect/HAL/IR",
        "@iree//iree/compiler/Dialect/IREE/IR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffects",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
    ],
)

cc_library(
    name = "tf_tensorlist_dialect",
    srcs = ["tf_tensorlist_dialect.cc"],
    hdrs = ["tf_tensorlist_dialect.h"],
    deps = [
        ":ir",
        "@llvm-project//mlir:IR",
    ],
)

gentbl_cc_library(
    name = "tf_tensorlist_ops_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tf_tensorlist_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tf_tensorlist_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_tensorlist_ops.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        ":td_files",
        "@iree//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)
