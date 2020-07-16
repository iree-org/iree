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

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "ModelBuilder",
    srcs = ["ModelBuilder.cpp"],
    hdrs = ["ModelBuilder.h"],
    deps = [
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:VectorOps",
    ],
)

cc_library(
    name = "VulkanLaunchWrapper",
    srcs = ["VulkanWrapperPass.cpp"],
    hdrs = ["VulkanWrapperPass.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ConversionPassIncGen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ModelRunner",
    srcs = ["ModelRunner.cpp"],
    hdrs = [
        "MemRefUtils.h",
        "ModelRunner.h",
    ],
    linkopts = [
        "-ldl",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:GPUToSPIRVTransforms",
        "@llvm-project//mlir:GPUToVulkanTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:StandardToSPIRVConversions",
        "@llvm-project//mlir:TargetLLVMIR",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorToSCF",
        "@llvm-project//mlir:mlir_c_runner_utils",
    ],
)
