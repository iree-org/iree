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

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "conversion",
    srcs = [
        "convert_tf_strings_to_strings.cc",
        "convert_tf_to_tf_strings.cc",
    ],
    hdrs = [
        "convert_tf_strings_to_strings.h",
        "convert_tf_to_tf_strings.h",
    ],
    deps = [
        "//integrations/tensorflow/compiler/dialect/tf_strings/ir",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Modules/Strings/IR",
        "//iree/compiler/Dialect/Modules/Strings/IR:Dialect",
        "//iree/compiler/Dialect/VM/Conversion",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardToStandard",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
    ],
    alwayslink = 1,
)
