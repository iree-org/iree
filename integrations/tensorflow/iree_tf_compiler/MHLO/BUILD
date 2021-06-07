# Copyright 2021 Google LLC
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
    name = "MHLO",
    srcs = [
        "EmitDefaultIREEABI.cpp",
        "FlattenTuplesInCFG.cpp",
        "Passes.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    defines = [
        "IREE_COMPILER_TENSORFLOW_ENABLED",
    ],
    deps = [
        "@iree//iree/compiler/Conversion:PassHeaders",
        "@iree//iree/compiler/Conversion/HLOToLinalg",
        "@iree//iree/compiler/Dialect/Flow/IR",
        "@iree//iree/compiler/Dialect/Flow/Transforms",
        "@iree//iree/compiler/Dialect/IREE/IR",
        "@iree//iree/compiler/Dialect/Shape/Transforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo:all_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo:chlo_legalize_to_hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo:mhlo_to_mhlo_lowering_patterns",
    ],
)
