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

load("//build_tools/bazel:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl(
    name = "FoldTensorExtractOpIncGen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "FoldTensorExtractOp.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FoldTensorExtractOp.td",
    td_srcs = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:MemRefOpsTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

cc_library(
    name = "LinalgToLLVM",
    srcs = [
        "ConvertToLLVM.cpp",
        "FoldTensorExtractOpPass.cpp",
        "KernelDispatch.cpp",
        "LLVMCodeGenOptions.cpp",
        "LinalgTileAndVectorizePass.cpp",
        "LinalgVectorizePass.cpp",
        "MaterializeCPULaunchConfigurationPass.cpp",
        "PadLinalgWorkgroupTiles.cpp",
        "Passes.cpp",
        "PlanConvLoopOrder.cpp",
        "UnfuseFMAOps.cpp",
    ],
    hdrs = [
        "KernelDispatch.h",
        "LLVMCodeGenOptions.h",
        "Passes.h",
    ],
    deps = [
        "//iree/compiler/Conversion/CodegenUtils",
        "//iree/compiler/Conversion/Common",
        "//iree/compiler/Conversion/HLOToHLO",
        "//iree/compiler/Conversion/LinalgToLLVM:FoldTensorExtractOpIncGen",
        "//iree/compiler/Conversion/VectorToLLVM",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/HAL/IR:HALDialect",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Shape/Transforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:CFGTransforms",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgInterfaces",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:StandardToSPIRV",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TosaToStandard",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorToSCF",
    ],
)
