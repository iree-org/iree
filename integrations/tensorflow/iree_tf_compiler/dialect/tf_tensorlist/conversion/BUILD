# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@iree//build_tools/bazel:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "convert_tf_to_tf_tensorlist",
    srcs = [
        "convert_tf_to_tf_tensorlist.cc",
    ],
    hdrs = [
        "convert_tf_to_tf_tensorlist.h",
    ],
    deps = [
        ":convert_tf_to_tf_tensorlist_rewriter_gen",
        "//iree_tf_compiler/dialect/tf_tensorlist/ir:tf_tensorlist_dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
    ],
)

gentbl_cc_library(
    name = "convert_tf_to_tf_tensorlist_rewriter_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "convert_tf_to_tf_tensorlist.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "convert_tf_to_tf_tensorlist.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "//iree_tf_compiler/dialect/tf_tensorlist/ir:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@iree//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
    ],
)

cc_library(
    name = "convert_tf_tensorlist_to_tensorlist",
    srcs = [
        "convert_tf_tensorlist_to_tensorlist.cc",
    ],
    hdrs = [
        "convert_tf_tensorlist_to_tensorlist.h",
    ],
    deps = [
        "//iree_tf_compiler/dialect/tf_tensorlist/ir",
        "//iree_tf_compiler/dialect/tf_tensorlist/ir:tf_tensorlist_dialect",
        "//iree_tf_compiler/dialect/utils:conversion_utils",
        "@iree//iree/compiler/Dialect/Modules/TensorList/IR",
        "@iree//iree/compiler/Dialect/Modules/TensorList/IR:TensorListDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
)
