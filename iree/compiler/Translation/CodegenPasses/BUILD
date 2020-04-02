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

# Utilities for working with IREE MLIR types.
load("//build_tools/bazel:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl(
    name = "LinalgVectorTransformPatternsIncGen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "LinalgVectorTransformPatterns.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LinalgVectorTransformPatterns.td",
    td_srcs = [
        "@llvm-project//mlir:LinalgTransformPatternsTdFiles",
    ],
)

cc_library(
    name = "CodegenPasses",
    srcs = [
        "HALInterfaceToMemrefPass.cpp",
        "HLOToLinalgOnBuffers.cpp",
        "HLOToLinalgOnTensors.cpp",
        "LinalgFusion.cpp",
        "LinalgVectorTransform.cpp",
        "Passes.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":LinalgVectorTransformPatternsIncGen",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Translation/CodegenUtils",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:map_xla_to_scalar_op",
    ],
    alwayslink = 1,
)
