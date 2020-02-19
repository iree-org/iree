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
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "LinalgToSPIRV",
    srcs = [
        "LinalgFusion.cpp",
        "LowerToSPIRV.cpp",
    ],
    hdrs = [
        "LowerToSPIRV.h",
        "Passes.h",
    ],
    deps = [
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Translation/CodegenUtils",
        "//iree/compiler/Translation/XLAToLinalg:IREELinalgTensorToBuffer",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AffineOps",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToSPIRVTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:LoopOps",
        "@llvm-project//mlir:LoopsToGPU",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVLowering",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardToSPIRVConversions",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_legalize_to_linalg",
    ],
    alwayslink = 1,
)
