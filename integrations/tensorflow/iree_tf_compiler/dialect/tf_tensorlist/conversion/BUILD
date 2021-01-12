# Copyright 2019 Google LLC
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

load("@iree//build_tools/bazel:tblgen.bzl", "gentbl")

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

gentbl(
    name = "convert_tf_to_tf_tensorlist_rewriter_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
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
