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
    name = "XLAToSPIRV",
    srcs = [
        "IREEToSPIRV.cpp",
        "IREEToSPIRVPass.cpp",
        "SPIRVLowering.cpp",
        "XLAToSPIRV.cpp",
    ],
    hdrs = [
        "IREEToSPIRV.h",
        "IREEToSPIRVPass.h",
        "SPIRVLowering.h",
        "TensorIndexToScalarValueMap.h",
        "XLAToSPIRV.h",
    ],
    deps = [
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Translation/CodegenUtils",
        "//iree/compiler/Translation/SPIRV/IndexComputation",
        "//iree/compiler/Translation/SPIRV/Passes",
        "//iree/compiler/Translation/SPIRV/ReductionCodegen",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AffineOps",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVLowering",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardToSPIRVConversions",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_legalize_to_standard",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_unfuse_batch_norm",
    ],
    alwayslink = 1,
)
