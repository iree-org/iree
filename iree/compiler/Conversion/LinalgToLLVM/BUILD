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
    name = "LinalgToLLVM",
    srcs = [
        "ConvertToLLVM.cpp",
        "HALInterfaceToMemrefArguments.cpp",
        "Passes.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//iree/compiler/Conversion/HLOToLinalg",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/HAL/IR:HALDialect",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Shape/Transforms",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:CFGTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorToSCF",
    ],
)
