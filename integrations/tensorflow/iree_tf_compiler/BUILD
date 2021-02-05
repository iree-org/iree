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

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "iree-tf-opt",
    srcs = ["iree-tf-opt-main.cpp"],
    deps = [
        "//iree_tf_compiler/TF",
        "//iree_tf_compiler/dialect/tf_strings/ir:dialect",
        "//iree_tf_compiler/dialect/tf_tensorlist/ir:tf_tensorlist_dialect",
        "@iree//iree/tools:init_xla_dialects",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_legalize_tf",
    ],
)

cc_binary(
    name = "iree-opt-tflite",
    srcs = ["iree-opt-tflite-main.cpp"],
    deps = [
        "//iree_tf_compiler/TFL",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
)

cc_binary(
    name = "iree-tf-import",
    srcs = ["iree-tf-import-main.cpp"],
    deps = [
        "//iree_tf_compiler/TF",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "@org_tensorflow//tensorflow/core/platform:errors",
    ] + select({
        "@org_tensorflow//tensorflow:oss": [],
        "//conditions:default": [
            # TODO: Enable this for all builds once dependency bug is fixed.
            "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",  # Note: always_links in random foo that makes constant folding work
        ],
    }),
)

cc_binary(
    name = "iree-import-tflite",
    srcs = ["iree-import-tflite-main.cpp"],
    deps = [
        "//iree_tf_compiler/TFL",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
    ],
)

cc_binary(
    name = "iree-import-xla",
    srcs = ["iree-import-xla-main.cpp"],
    deps = [
        "//iree_tf_compiler/TF",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo_to_mlir_hlo",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
