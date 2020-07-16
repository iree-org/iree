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

load("//iree:build_defs.oss.bzl", iree_cc_binary = "cc_binary")
load("//build_tools/embed_data:build_defs.bzl", "cc_embed_data")
load("//build_tools/bazel:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "dialect",
    srcs = [
        "conversion_patterns.cc",
        "custom_dialect.cc",
        "custom_ops.cc.inc",
    ],
    hdrs = [
        "conversion_patterns.h",
        "custom_dialect.h",
        "custom_ops.h.inc",
    ],
    deps = [
        ":custom_imports",
        ":custom_ops_gen",
        "//iree/compiler/Dialect/HAL/Conversion",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl(
    name = "custom_ops_gen",
    tbl_outs = [
        ("-gen-op-decls", "custom_ops.h.inc"),
        ("-gen-op-defs", "custom_ops.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "custom_ops.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/HAL/IR:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_embed_data(
    name = "custom_imports",
    srcs = ["custom.imports.mlir"],
    cc_file_output = "custom.imports.cc",
    cpp_namespace = "mlir::iree_compiler::IREE::Custom",
    flatten = True,
    h_file_output = "custom.imports.h",
)

cc_library(
    name = "init_dialect",
    hdrs = ["init_dialect.h"],
    deps = [
        ":dialect",
        "@llvm-project//mlir:IR",
    ],
)

iree_cc_binary(
    name = "custom-opt",
    srcs = ["custom_opt.cc"],
    deps = [
        ":init_dialect",
        "//iree/compiler/Conversion:init_conversions",
        "//iree/compiler/Conversion/HLOToLinalg",
        "//iree/compiler/Dialect/Flow/Analysis",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/Flow/Transforms",
        "//iree/compiler/Dialect/HAL/Conversion:Passes",
        "//iree/compiler/Dialect/HAL/Conversion/FlowToHAL",
        "//iree/compiler/Dialect/HAL/Conversion/HALToVM",
        "//iree/compiler/Dialect/HAL/Transforms",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/IREE/Transforms",
        "//iree/compiler/Dialect/Shape/Conversion",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Shape/Transforms",
        "//iree/compiler/Dialect/VM/Analysis",
        "//iree/compiler/Dialect/VM/Conversion/StandardToVM",
        "//iree/compiler/Dialect/VM/IR",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/compiler/Dialect/VMLA/Conversion/HLOToVMLA",
        "//iree/compiler/Dialect/VMLA/Conversion/VMLAToVM",
        "//iree/compiler/Dialect/VMLA/Transforms",
        "//iree/compiler/Dialect/Vulkan/IR",
        "//iree/compiler/Translation:IREEVM",
        "//iree/tools:init_compiler_modules",
        "//iree/tools:init_iree_passes_and_dialects",
        "//iree/tools:init_mlir_passes_and_dialects",
        "//iree/tools:init_targets",
        "//iree/tools:init_xla_dialects",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo:legalize_control_flow",
    ],
)

iree_cc_binary(
    name = "custom-translate",
    srcs = ["custom_translate.cc"],
    deps = [
        ":init_dialect",
        "//iree/compiler/Conversion:init_conversions",
        "//iree/compiler/Dialect/VM/Target:init_targets",
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/compiler/Translation:IREEVM",
        "//iree/tools:init_compiler_modules",
        "//iree/tools:init_iree_passes_and_dialects",
        "//iree/tools:init_mlir_passes_and_dialects",
        "//iree/tools:init_targets",
        "//iree/tools:init_translations",
        "//iree/tools:init_xla_dialects",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)
