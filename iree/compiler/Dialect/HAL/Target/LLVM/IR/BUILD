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

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_TARGET_BACKEND_LLVM-IR})
  return()
endif()
""",
)

cc_library(
    name = "LLVMIR",
    srcs = [
        "LLVMIRTarget.cpp",
    ],
    hdrs = [
        "LLVMIRTarget.h",
    ],
    deps = [
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Dialect/HAL/Target/LLVM:LLVMBaseTarget",
        "//iree/compiler/Dialect/HAL/Target/LLVM:LLVMIRPasses",
        "//iree/compiler/Dialect/HAL/Target/LLVM:LLVMTargetOptions",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/schemas:llvmir_executable_def_cc_fbs",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:TargetLLVMIR",
    ],
)
