# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//build_tools/bazel:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["tf_tensorlist_base.td"])

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
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
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/IREE/IR",
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
        "//integrations/tensorflow/compiler/dialect/tf_tensorlist/conversion:convert_flow_to_hal",
        "//iree/compiler/Dialect/IREE/IR",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

gentbl(
    name = "tf_tensorlist_ops_gen",
    tbl_outs = [
        ("-gen-op-decls", "tf_tensorlist_ops.h.inc"),
        ("-gen-op-defs", "tf_tensorlist_ops.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_tensorlist_ops.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)
