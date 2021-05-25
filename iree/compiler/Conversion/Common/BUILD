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

load("//build_tools/bazel:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl_cc_library(
    name = "FoldTensorExtractOpIncGen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
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
    name = "Common",
    srcs = [
        "BufferAllocViewCleanUpPass.cpp",
        "DemoteF32ToF16.cpp",
        "FlattenMemRefSubspanPass.cpp",
        "FoldTensorExtractOpPass.cpp",
        "ForOpCanonicalizationPass.cpp",
        "LaunchConfig.cpp",
        "LinalgBufferizePass.cpp",
        "Passes.cpp",
        "SetNumWorkgroupsPass.cpp",
        "Transforms.cpp",
        "VectorTransferOptimization.cpp",
    ],
    hdrs = [
        "LaunchConfig.h",
        "Passes.h",
        "Transforms.h",
    ],
    deps = [
        "//iree/compiler/Conversion:PassHeaders",
        "//iree/compiler/Conversion/CodegenUtils",
        "//iree/compiler/Conversion/Common:FoldTensorExtractOpIncGen",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:CFGTransforms",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgInterfaces",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
    ],
)
