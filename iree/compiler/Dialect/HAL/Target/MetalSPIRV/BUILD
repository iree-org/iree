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
    name = "MetalSPIRV",
    srcs = ["MetalSPIRVTarget.cpp"],
    hdrs = ["MetalSPIRVTarget.h"],
    deps = [
        ":SPIRVToMSL",
        "//iree/compiler/Conversion/Common",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Dialect/HAL/Target/SPIRVCommon",
        "//iree/schemas:metal_executable_def_c_fbs",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:VectorOps",
    ],
)

cc_library(
    name = "SPIRVToMSL",
    srcs = [
        "SPIRVToMSL.cpp",
    ],
    hdrs = ["SPIRVToMSL.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@spirv_cross//:spirv_cross_lib",
    ],
)
