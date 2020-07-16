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
        "CleanupPlaceholdersPass.cpp",
        "ConvertHLOToShapeDialectPass.cpp",
        "FunctionSignatureExpansionPass.cpp",
        "HoistShapeCalculationsPass.cpp",
        "MaterializeShapeCalculations.cpp",
        "MaterializeShapeCalculationsPass.cpp",
        "TieDynamicShapesPass.cpp",
    ],
    hdrs = [
        "Passes.h",
        "Patterns.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Shape/Plugins/VMLA:VMLAShapeBuilder",
        "//iree/compiler/Dialect/Shape/Plugins/XLA:XlaHloShapeBuilder",
        "//iree/compiler/Dialect/Shape/Utils:TypeConversion",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo",
    ],
)
