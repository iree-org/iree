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

# Transforms specific to the IREE Strings module.

load("//build_tools/bazel:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "conversion",
    deps = [
        ":convert_flow_to_hal",
        ":convert_tf_to_tf_strings",
    ],
)

cc_library(
    name = "convert_flow_to_hal",
    srcs = [
        "convert_flow_to_hal.cc",
    ],
    hdrs = [
        "convert_flow_to_hal.h",
    ],
    deps = [
        "//integrations/tensorflow/compiler/dialect/tf_strings/ir:ops",
        "//iree/compiler/Dialect/HAL/Conversion",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Modules/Strings/IR",
        "//iree/compiler/Dialect/Modules/Strings/IR:Dialect",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

gentbl(
    name = "convert_tf_to_tf_strings_rewriter_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "convert_tf_to_tf_strings.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "convert_tf_to_tf_strings.td",
    td_srcs = [
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "//integrations/tensorflow/compiler/dialect/tf_strings/ir:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
    ],
)

cc_library(
    name = "convert_tf_to_tf_strings",
    srcs = [
        "convert_tf_to_tf_strings.cc",
    ],
    hdrs = [
        "convert_tf_to_tf_strings.h",
    ],
    deps = [
        ":convert_tf_to_tf_strings_rewriter_gen",
        "//integrations/tensorflow/compiler/dialect/tf_strings/ir",
        "//integrations/tensorflow/compiler/dialect/tf_strings/ir:dialect",
        "//integrations/tensorflow/compiler/dialect/tf_strings/ir:ops",
        "//iree/compiler/Utils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
    ],
    alwayslink = 1,
)
