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
#
load("//iree:build_defs.oss.bzl", "platform_trampoline_deps")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
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
        ":LLVMIRPasses",
        ":LLVMTargetOptions",
        "//iree/compiler/Conversion/LinalgToLLVM",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/schemas:llvmir_executable_def_cc_fbs",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        # TODO(ataei): Link with native target dep.
        "@llvm-project//llvm:X86CodeGen",
        "@llvm-project//mlir:TargetLLVMIR",
    ],
)

cc_library(
    name = "LLVMAOT",
    srcs = [
        "LLVMAOTTarget.cpp",
    ],
    hdrs = [
        "LLVMAOTTarget.h",
    ],
    deps = [
        ":LLVMAOTTargetLinker",
        ":LLVMIRPasses",
        ":LLVMTargetOptions",
        "//iree/compiler/Conversion/LinalgToLLVM",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/schemas:dylib_executable_def_cc_fbs",
        "@llvm-project//llvm:AArch64AsmParser",
        "@llvm-project//llvm:AArch64CodeGen",
        "@llvm-project//llvm:ARMAsmParser",
        "@llvm-project//llvm:ARMCodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:X86AsmParser",
        "@llvm-project//llvm:X86CodeGen",
        "@llvm-project//mlir:TargetLLVMIR",
    ],
)

cc_library(
    name = "LLVMIRPasses",
    srcs = [
        "LLVMIRPasses.cpp",
    ],
    hdrs = [
        "LLVMIRPasses.h",
    ],
    deps = [
        ":LLVMTargetOptions",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "LLVMTargetOptions",
    srcs = [
        "LLVMTargetOptions.cpp",
    ],
    hdrs = [
        "LLVMTargetOptions.h",
    ],
    deps = [
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
    ],
)

cc_library(
    name = "LLVMAOTTargetLinker",
    hdrs = ["LLVMAOTTargetLinker.h"],
    deps = [
        "//iree/base:file_io",
    ] + platform_trampoline_deps("LLVMAOTTargetLinker", "compiler/Dialect/HAL/Target/LLVM"),
)

cc_library(
    name = "LLVMAOTTargetLinker_hdrs",
    hdrs = ["LLVMAOTTargetLinker.h"],
    deps = [
        ":LLVMTargetOptions",
        "//iree/base:file_io",
    ],
)
