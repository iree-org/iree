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

cc_library(
    name = "TF",
    srcs = [
        "ConvertToMHLO.cpp",
        "LowerExportedFunctions.cpp",
        "LowerGlobalTensors.cpp",
        "Passes.cpp",
        "PropagateResourceCasts.cpp",
        "StripMetadata.cpp",
        "VerifyFullyConverted.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    defines = [
        "IREE_COMPILER_TENSORFLOW_ENABLED",
    ],
    deps = [
        "//iree_tf_compiler/dialect/tf_strings/conversion:convert_tf_strings_to_strings",
        "//iree_tf_compiler/dialect/tf_strings/conversion:convert_tf_to_tf_strings",
        "//iree_tf_compiler/dialect/tf_strings/ir:dialect",
        "//iree_tf_compiler/dialect/tf_tensorlist/conversion:convert_tf_tensorlist_to_tensorlist",
        "//iree_tf_compiler/dialect/tf_tensorlist/conversion:convert_tf_to_tf_tensorlist",
        "//iree_tf_compiler/dialect/tf_tensorlist/ir:tf_tensorlist_dialect",
        "@iree//iree/base:signature_mangle",
        "@iree//iree/compiler/Dialect/Flow/IR",
        "@iree//iree/compiler/Dialect/IREE/IR",
        "@iree//iree/compiler/Dialect/Shape/Conversion",
        "@iree//iree/compiler/Dialect/Shape/Transforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo:chlo_legalize_to_hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_legalize_tf",
    ],
)
