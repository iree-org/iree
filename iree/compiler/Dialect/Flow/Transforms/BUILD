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
    name = "Transforms",
    srcs = [
        "CreateBenchmarkFuncs.cpp",
        "DeduplicateExecutables.cpp",
        "DestructiveUpdateUtils.cpp",
        "DispatchConfig.cpp",
        "DispatchLinalgOnTensors.cpp",
        "DispatchabilityAnalysis.cpp",
        "FlattenTuplesInCFG.cpp",
        "FoldCompatibleDispatchRegions.cpp",
        "FormStreams.cpp",
        "HLOToHLOPreprocessing.cpp",
        "HoistUnstreamableOps.cpp",
        "IdentifyDispatchRegions.cpp",
        "IdentifyDispatchRegions2.cpp",
        "InjectDispatchTracing.cpp",
        "LegalizeInputTypes.cpp",
        "MaterializeExportedReflection.cpp",
        "MergeExportedReflection.cpp",
        "OutlineDispatchRegions.cpp",
        "OutlineDispatchRegions2.cpp",
        "OutlineLargeConstants.cpp",
        "Passes.cpp",
        "PrePostPartitioningConversion.cpp",
        "RematerializeDispatchConstants.cpp",
        "StripAndSplatConstantVariables.cpp",
    ],
    hdrs = [
        "DestructiveUpdateUtils.h",
        "DispatchConfig.h",
        "Passes.h",
    ],
    deps = [
        "//iree/base:signature_mangle",
        "//iree/compiler/Conversion/HLOToLinalg",
        "//iree/compiler/Conversion/HLOToLinalg:HLOToLinalgOnTensors",
        "//iree/compiler/Dialect/Flow/Analysis",
        "//iree/compiler/Dialect/Flow/Conversion",
        "//iree/compiler/Dialect/Flow/Conversion/HLOToFlow",
        "//iree/compiler/Dialect/Flow/Conversion/StandardToFlow",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/Flow/Utils",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/Conversion",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Shape/Transforms",
        "//iree/compiler/Dialect/Shape/Utils:TypeConversion",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@mlir-hlo//:chlo_legalize_to_hlo",
        "@mlir-hlo//:hlo",
        "@mlir-hlo//:legalize_control_flow",
        "@mlir-hlo//:legalize_gather_to_torch_index_select",
        "@mlir-hlo//:materialize_broadcasts",
        "@mlir-hlo//:mhlo_to_mhlo_lowering_patterns",
        "@mlir-hlo//:unfuse_batch_norm",
    ],
)
