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
    name = "Transforms",
    srcs = [
        "Conversion.cpp",
        "Passes.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//iree/compiler/Conversion/Common",
        "//iree/compiler/Conversion/LinalgToLLVM",
        "//iree/compiler/Dialect/HAL/IR:HALDialect",
        "//iree/compiler/Dialect/HAL/Transforms",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/IREE/Transforms",
        "//iree/compiler/Dialect/Modules/VMVX/Conversion/HALToVMVX",
        "//iree/compiler/Dialect/Modules/VMVX/Conversion/StandardToVMVX",
        "//iree/compiler/Dialect/Modules/VMVX/IR",
        "//iree/compiler/Dialect/Modules/VMVX/IR:VMVXDialect",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Shape/Transforms",
        "//iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:CFGTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgInterfaces",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TosaToStandard",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorToSCF",
    ],
)
