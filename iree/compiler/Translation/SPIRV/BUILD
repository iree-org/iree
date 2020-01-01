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
    name = "SPIRV",
    srcs = [
        "AdjustIntegerWidthPass.cpp",
        "EmbeddedKernels.cpp",
        "IREEIndexComputation.cpp",
        "IREEToSPIRV.cpp",
        "IREEToSPIRVPass.cpp",
        "IndexComputation.cpp",
        "IndexComputationAttribute.cpp",
        "IndexComputationPass.cpp",
        "PrepareReductionDispatch.cpp",
        "ReductionFnLowering.cpp",
        "SPIRVLowering.cpp",
        "SPIRVLowering.h",
        "XLAIndexPropagation.cpp",
        "XLAToSPIRV.cpp",
    ],
    hdrs = [
        "EmbeddedKernels.h",
        "IREECodegenUtils.h",
        "IREEIndexComputation.h",
        "IREEToSPIRV.h",
        "IREEToSPIRVPass.h",
        "IndexComputation.h",
        "IndexComputationAttribute.h",
        "TensorIndexToScalarValueMap.h",
        "XLAIndexPropagation.h",
        "XLAToSPIRV.h",
    ],
    deps = [
        "//iree/compiler/IR",
        "//iree/compiler/Translation/SPIRV/Kernels",
        "//iree/compiler/Utils",
        "//iree/schemas:spirv_executable_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVDialectRegistration",
        "@llvm-project//mlir:SPIRVLowering",
        "@llvm-project//mlir:SPIRVSerialization",
        "@llvm-project//mlir:StandardDialectRegistration",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:Translation",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
    ],
    alwayslink = 1,
)
