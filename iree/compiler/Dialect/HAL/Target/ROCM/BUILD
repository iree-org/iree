# Copyright 2021 Google LLC
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

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT "${IREE_TARGET_BACKEND_ROCM}")
  return()
endif()
""",
)

cc_library(
    name = "ROCM",
    srcs = [
        "ROCMTarget.cpp",
        "ROCMTargetUtils.cpp",
    ],
    hdrs = [
        "ROCMTarget.h",
    ],
    deps = [
        "//iree/base/internal:flatcc",
        "//iree/compiler/Conversion/LinalgToLLVMGPU",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Utils",
        "//iree/schemas:rocm_executable_def_c_fbs",
        "@llvm-project//llvm:AMDGPUCodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)
