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
        "AffineExprCodegen.h",
        "EmbeddedKernels.cpp",
        "IREEIndexComputation.cpp",
        "IREEToSPIRV.cpp",
        "IREEToSPIRVPass.cpp",
        "IndexComputation.cpp",
        "IndexComputationAttribute.cpp",
        "IndexComputationPass.cpp",
        "SPIRVExecutableTranslation.cpp",
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
        "SPIRVExecutableTranslation.h",
        "XLAIndexPropagation.h",
        "XLAToSPIRV.h",
    ],
    deps = [
        "//iree/compiler/IR",
        "//iree/compiler/Translation/SPIRV/Kernels",
        "//iree/compiler/Utils",
        "//iree/schemas",
        "//iree/schemas:spirv_executable_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:SPIRVDialect",
        "@local_config_mlir//:SPIRVDialectRegistration",
        "@local_config_mlir//:SPIRVSerialization",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:Transforms",
        "@local_config_mlir//:Translation",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
    ],
    alwayslink = 1,
)
