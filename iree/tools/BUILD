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
# Compiler tooling, like the compiler, is not designed to run on device and is tagged as "hostonly".

load(
    "//iree:build_defs.oss.bzl",
    "IREE_DRIVER_MODULES",
    "PLATFORM_VULKAN_DEPS",
    iree_cc_binary = "cc_binary",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "run_lit.sh",
    "sanitizer_suppressions.txt",
])

iree_cc_binary(
    name = "iree-benchmark-module",
    testonly = True,
    srcs = ["iree-benchmark-module-main.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
        "//iree/base:init",
        "//iree/base:file_io",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/modules/hal",
        "//iree/tools/utils:vm_util",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ] + PLATFORM_VULKAN_DEPS + IREE_DRIVER_MODULES,
)

iree_cc_binary(
    name = "iree-dump-module",
    srcs = ["iree-dump-module-main.cc"],
    deps = [
        "//iree/base:file_io",  # build-cleaner: keep
        "//iree/base:flatbuffer_util",
        "//iree/base:init",
        "//iree/schemas:bytecode_module_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
    ],
)

cc_library(
    name = "init_iree_passes_and_dialects",
    hdrs = [
        "init_iree_dialects.h",
        "init_iree_passes.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Flow/Analysis",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/Flow/Transforms",
        "//iree/compiler/Dialect/HAL/IR:HALDialect",
        "//iree/compiler/Dialect/HAL/Transforms",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/IREE/Transforms",
        "//iree/compiler/Dialect/Sequence/IR",
        "//iree/compiler/Dialect/Shape/Conversion",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Shape/Transforms",
        "//iree/compiler/Dialect/VM/Analysis",
        "//iree/compiler/Dialect/VM/IR",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/compiler/Dialect/VMLA/IR:VMLADialect",
        "//iree/compiler/Dialect/VMLA/Transforms",
        "//iree/compiler/Dialect/Vulkan/IR",
        "//iree/compiler/Translation:IREEVM",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "init_mlir_passes_and_dialects",
    hdrs = [
        "init_mlir_dialects.h",
        "init_mlir_passes.h",
    ],
    deps = [
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToSPIRVTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgPassIncGen",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgToSPIRV",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToGPUPass",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:SDBM",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVLowering",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardToSPIRVConversions",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
    ],
)

cc_library(
    name = "init_passes_and_dialects",
    hdrs = [
        "init_dialects.h",
        "init_passes.h",
    ],
    deps = [
        ":init_compiler_modules",
        ":init_iree_passes_and_dialects",
        ":init_mlir_passes_and_dialects",
        ":init_xla_dialects",
        "//iree/compiler/Conversion:init_conversions",
        "//iree/compiler/Conversion/HLOToLinalg",
        "//iree/compiler/Dialect/HAL/Conversion:Passes",
    ],
)

cc_library(
    name = "init_compiler_modules",
    hdrs = ["init_compiler_modules.h"],
    deps = [
        "//iree/compiler/Dialect/Modules/Check/IR:CheckDialect",
        "//iree/compiler/Dialect/Modules/Strings/IR:Dialect",
        "//iree/compiler/Dialect/Modules/TensorList/IR:TensorListDialect",
    ],
)

cc_library(
    name = "init_targets",
    srcs = ["init_targets.cc"],
    hdrs = ["init_targets.h"],
    local_defines = [
        "IREE_HAVE_LLVMAOT_TARGET",
        "IREE_HAVE_LLVMIR_TARGET",
        "IREE_HAVE_VMLA_TARGET",
        "IREE_HAVE_VULKANSPIRV_TARGET",
    ],
    deps = [
        "//iree/compiler/Dialect/HAL/Target/LLVM/AOT:LLVMAOT",
        "//iree/compiler/Dialect/HAL/Target/LLVM/IR:LLVMIR",
        "//iree/compiler/Dialect/HAL/Target/VMLA",
        "//iree/compiler/Dialect/HAL/Target/VulkanSPIRV",
    ],
)

cc_library(
    name = "init_translations",
    hdrs = [
        "init_translations.h",
    ],
    deps = [
        "//iree/compiler/Translation:IREEVM",
        "@llvm-project//mlir:SPIRVTranslateRegistration",
    ],
)

cc_library(
    name = "init_xla_dialects",
    hdrs = [
        "init_xla_dialects.h",
    ],
    deps = [
        "@llvm-project//mlir:IR",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo:lhlo",
    ],
)

cc_library(
    name = "iree_opt_main",
    srcs = ["iree-opt-main.cc"],
    deps = [
        ":init_passes_and_dialects",
        ":init_targets",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Support",
    ],
)

iree_cc_binary(
    name = "iree-opt",
    tags = ["hostonly"],
    deps = [
        ":iree_opt_main",
    ],
)

iree_cc_binary(
    name = "iree-run-mlir",
    srcs = ["iree-run-mlir-main.cc"],
    tags = ["hostonly"],
    deps = [
        ":init_passes_and_dialects",
        ":init_targets",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//iree/base:api",
        "//iree/base:tracing",
        "//iree/compiler/Dialect/Flow/Transforms",
        "//iree/compiler/Dialect/HAL/Transforms",
        "//iree/compiler/Dialect/IREE/Transforms",
        "//iree/compiler/Dialect/VM/Target:init_targets",
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/hal:api",
        "//iree/modules/hal",
        "//iree/tools/utils:vm_util",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/compiler/Translation:IREEVM",
    ] + PLATFORM_VULKAN_DEPS + IREE_DRIVER_MODULES,
)

iree_cc_binary(
    name = "iree-run-module",
    srcs = ["iree-run-module-main.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "//iree/base:file_io",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/modules/hal",
        "//iree/tools/utils:vm_util",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ] + PLATFORM_VULKAN_DEPS + IREE_DRIVER_MODULES,
)

iree_cc_binary(
    name = "iree-tblgen",
    tags = ["hostonly"],
    deps = [
        "//iree/compiler/Dialect/IREE/Tools",
        "//iree/compiler/Dialect/VM/Tools",
        "@llvm-project//mlir:MlirTableGenMain",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TableGen",
    ],
)

cc_library(
    name = "iree_translate_main",
    srcs = ["iree-translate-main.cc"],
    deps = [
        ":init_compiler_modules",
        ":init_iree_passes_and_dialects",
        ":init_mlir_passes_and_dialects",
        ":init_targets",
        ":init_translations",
        ":init_xla_dialects",
        "//iree/compiler/Conversion:init_conversions",
        "//iree/compiler/Dialect/VM/Target:init_targets",
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/compiler/Translation:IREEVM",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)

iree_cc_binary(
    name = "iree-translate",
    tags = ["hostonly"],
    deps = [
        ":iree_translate_main",
    ],
)

sh_binary(
    name = "IreeFileCheck",
    testonly = True,
    srcs = ["IreeFileCheck.sh"],
    data = ["@llvm-project//llvm:FileCheck"],
    tags = ["hostonly"],
)
