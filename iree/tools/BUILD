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

# Misc tools used to optimize, translate, and evaluate IREE.
# Most of these are not designed to run on-device.

load("//iree:build_defs.oss.bzl", "IREE_DRIVER_MODULES", "PLATFORM_VULKAN_DEPS")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "run_lit.sh",
    "sanitizer_suppressions.txt",
])

TARGET_COMPILER_BACKENDS = [
    "//iree/compiler/Dialect/HAL/Target/LegacyInterpreter",
    "//iree/compiler/Dialect/HAL/Target/VMLA",
    "//iree/compiler/Dialect/HAL/Target/VulkanSPIRV",
    "//iree/compiler/Dialect/HAL/Target/LLVM",
]

cc_binary(
    name = "iree-benchmark-module",
    testonly = True,
    srcs = ["benchmark_module_main.cc"],
    deps = [
        ":vm_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
        "//iree/base:api_util",
        "//iree/base:file_io",
        "//iree/base:localfile",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/modules/hal",
        "//iree/testing:benchmark_main",
        "//iree/vm:bytecode_module",
    ] + PLATFORM_VULKAN_DEPS + IREE_DRIVER_MODULES,
)

cc_binary(
    name = "iree-dump-module",
    srcs = ["dump_module_main.cc"],
    deps = [
        "//iree/base:file_io",  # build-cleaner: keep
        "//iree/base:flatbuffer_util",
        "//iree/base:init",
        "//iree/base:localfile",
        "//iree/schemas:bytecode_module_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
    ],
)

cc_library(
    name = "iree_opt_library",
    deps = [
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/IREE/Transforms",
        "//iree/compiler/Dialect/Flow/Analysis",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/Flow/Transforms",
        "//iree/compiler/Dialect/HAL/Conversion/FlowToHAL",
        "//iree/compiler/Dialect/HAL/Conversion/HALToVM",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/HAL/Transforms",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Shape/Transforms",
        "//iree/compiler/Dialect/VM/Analysis",
        "//iree/compiler/Dialect/VM/Conversion/StandardToVM",
        "//iree/compiler/Dialect/VM/IR",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/compiler/Dialect/VMLA/Conversion/HLOToVMLA",
        "//iree/compiler/Dialect/VMLA/Conversion/VMLAToVM",
        "//iree/compiler/Dialect/VMLA/IR",
        "//iree/compiler/Dialect/VMLA/Transforms",
        "//iree/compiler/Dialect/Vulkan/IR",
        "//iree/compiler/Translation/Interpreter/Transforms",
        "//iree/compiler/Translation:IREEVM",
        "//iree/compiler/Translation/XLAToLinalg",
        "//iree/compiler/Translation/XLAToLinalg:ReductionLowering",
        "//iree/compiler/Translation/SPIRV/XLAToSPIRV",
        "//iree/compiler/Translation/SPIRV/LinalgToSPIRV",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_dialect_registration",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_legalize_control_flow",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_test_passes",
    ] + TARGET_COMPILER_BACKENDS,
    alwayslink = 1,
)

cc_binary(
    name = "iree-opt",
    deps = [
        ":iree_opt_library",
        "@llvm-project//mlir:MlirOptMain",
    ],
)

cc_binary(
    name = "iree-run-mlir",
    srcs = ["run_mlir_main.cc"],
    deps = [
        ":vm_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//iree/base:api",
        "//iree/base:api_util",
        "//iree/base:source_location",
        "//iree/compiler/Dialect/Flow/Transforms",
        "//iree/compiler/Dialect/HAL/Transforms",
        "//iree/compiler/Dialect/IREE/Transforms",
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/hal:api",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:value",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopOpsTransforms",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/compiler/Translation:IREEVM",
    ] + TARGET_COMPILER_BACKENDS + PLATFORM_VULKAN_DEPS + IREE_DRIVER_MODULES,
)

cc_binary(
    name = "iree-run-module",
    srcs = ["run_module_main.cc"],
    deps = [
        ":vm_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "//iree/base:api_util",
        "//iree/base:file_io",
        "//iree/base:init",
        "//iree/base:localfile",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/modules/hal",
        "//iree/vm:bytecode_module",
    ] + PLATFORM_VULKAN_DEPS + IREE_DRIVER_MODULES,
)

cc_binary(
    name = "iree-tblgen",
    linkopts = [
        "-lpthread",
    ],
    deps = [
        "//iree/compiler/Dialect/VM/Tools",
        "@llvm-project//mlir:MlirTableGenMain",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TableGen",
    ],
)

cc_library(
    name = "iree_translate_library",
    srcs = ["translate_main.cc"],
    deps = [
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/compiler/Translation:IREEVM",
        "//iree/compiler/Translation/SPIRV/XLAToSPIRV",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopOpsTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateClParser",
        "@llvm-project//mlir:Translation",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_dialect_registration",
    ] + TARGET_COMPILER_BACKENDS,
    alwayslink = 1,
)

cc_binary(
    name = "iree-translate",
    deps = [
        ":iree_translate_library",
    ],
)

sh_binary(
    name = "IreeFileCheck",
    testonly = True,
    srcs = ["IreeFileCheck.sh"],
    data = ["@llvm-project//llvm:FileCheck"],
)

# TODO(b/146898896): Refactor these into more coherent packages.
cc_library(
    name = "vm_util",
    srcs = ["vm_util.cc"],
    hdrs = ["vm_util.h"],
    deps = [
        "//iree/base:api_util",
        "//iree/base:buffer_string_util",
        "//iree/base:shape",
        "//iree/base:shaped_buffer",
        "//iree/base:shaped_buffer_string_util",
        "//iree/base:signature_mangle",
        "//iree/base:status",
        "//iree/hal:api",
        "//iree/modules/hal",
        "//iree/vm:bytecode_module",
        "//iree/vm:module",
        "//iree/vm:variant_list",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "vm_util_test",
    srcs = ["vm_util_test.cc"],
    deps = [
        ":vm_util",
        "//iree/base:api",
        "//iree/base:status_matchers",
        "//iree/hal:api",
        "//iree/hal/interpreter:interpreter_driver_module",
        "//iree/modules/hal",
        "//iree/testing:gtest_main",
        "//iree/vm:value",
        "//iree/vm:variant_list",
    ],
)
