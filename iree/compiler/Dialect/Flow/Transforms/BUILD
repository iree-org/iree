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
    name = "Transforms",
    srcs = [
        "AssignExecutableWorkloads.cpp",
        "DispatchabilityAnalysis.cpp",
        "FlattenTuplesInCFG.cpp",
        "FoldCompatibleDispatchRegions.cpp",
        "FormStreams.cpp",
        "IdentifyDispatchRegions.cpp",
        "IdentifyReductionRegions.cpp",
        "LegalizeInputTypes.cpp",
        "MaterializeExportedReflection.cpp",
        "MergeExportedReflection.cpp",
        "OutlineDispatchRegions.cpp",
        "OutlineReductionRegions.cpp",
        "Passes.cpp",
        "PrePostPartitioningConversion.cpp",
        "RematerializeDispatchConstants.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//iree/base:signature_mangle",
        "//iree/compiler/Dialect/Flow/Analysis",
        "//iree/compiler/Dialect/Flow/Conversion",
        "//iree/compiler/Dialect/Flow/Conversion/HLOToFlow",
        "//iree/compiler/Dialect/Flow/Conversion/StandardToFlow",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/Flow/Utils",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_legalize_control_flow",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_lower",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_materialize_broadcasts",
    ],
    alwayslink = 1,
)
