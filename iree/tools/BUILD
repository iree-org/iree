# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Misc tools used to optimize, translate, and evaluate IREE.
# Compiler tooling, like the compiler, is not designed to run on device and is tagged as "hostonly".

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "run_lit.sh",
])

cc_binary(
    name = "iree-benchmark-module",
    srcs = ["iree-benchmark-module-main.cc"],
    deps = [
        "//iree/base",
        "//iree/base:cc",
        "//iree/base:tracing",
        "//iree/base/internal:file_io",
        "//iree/base/internal:flags",
        "//iree/hal",
        "//iree/hal/drivers",
        "//iree/modules/hal",
        "//iree/tools/utils:vm_util",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:cc",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "iree-benchmark-trace",
    srcs = ["iree-benchmark-trace-main.c"],
    deps = [
        "//iree/base",
        "//iree/base:tracing",
        "//iree/base/internal:atomic_slist",
        "//iree/base/internal:file_path",
        "//iree/base/internal:flags",
        "//iree/hal",
        "//iree/hal/drivers",
        "//iree/testing:benchmark",
        "//iree/tools/utils:trace_replay",
        "//iree/tools/utils:yaml_util",
        "//iree/vm",
        "@com_github_yaml_libyaml//:yaml",
    ],
)

cc_binary(
    name = "iree-check-module",
    testonly = True,
    srcs = ["iree-check-module-main.cc"],
    deps = [
        "//iree/base",
        "//iree/base:cc",
        "//iree/base:core_headers",
        "//iree/base:logging",
        "//iree/base:tracing",
        "//iree/base/internal:file_io",
        "//iree/base/internal:flags",
        "//iree/hal",
        "//iree/hal/drivers",
        "//iree/modules/check",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/tools/utils:vm_util",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)

cc_binary(
    name = "iree-dump-module",
    srcs = ["iree-dump-module-main.cc"],
    deps = [
        "//iree/base",
        "//iree/base/internal:file_io",
        "//iree/base/internal/flatcc:debugging",
        "//iree/schemas:bytecode_module_def_c_fbs",
        "//iree/tools/utils:vm_util",
    ],
)

cc_library(
    name = "init_iree_passes_and_dialects",
    hdrs = [
        "init_iree_dialects.h",
        "init_iree_passes.h",
    ],
    deps = [
        "//iree/compiler/Bindings/Native/Transforms",
        "//iree/compiler/Bindings/TFLite/Transforms",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/Flow/Transforms",
        "//iree/compiler/Dialect/HAL/IR:HALDialect",
        "//iree/compiler/Dialect/HAL/Transforms",
        "//iree/compiler/Dialect/LinalgExt/IR",
        "//iree/compiler/Dialect/LinalgExt/IR:TiledOpInterface",
        "//iree/compiler/Dialect/LinalgExt/Transforms",
        "//iree/compiler/Dialect/Modules/VMVX/IR:VMVXDialect",
        "//iree/compiler/Dialect/Modules/VMVX/Transforms",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Shape/Transforms",
        "//iree/compiler/Dialect/Stream/IR",
        "//iree/compiler/Dialect/Stream/Transforms",
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/Util/Transforms",
        "//iree/compiler/Dialect/VM/Analysis",
        "//iree/compiler/Dialect/VM/IR",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/compiler/Dialect/Vulkan/IR",
        "//iree/compiler/InputConversion/Common",
        "//iree/compiler/InputConversion/MHLO",
        "//iree/compiler/InputConversion/TOSA",
        "//iree/compiler/Translation:IREEVM",
        "//llvm-external-projects/iree-dialects:IREEDialect",
        "//llvm-external-projects/iree-dialects:IREEPyDMDialect",
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
        "@llvm-project//mlir:GPUToSPIRV",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgPassIncGen",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgToSPIRV",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToGPUPass",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVTransforms",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardToSPIRV",
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
        "//iree/compiler/Codegen",
        "//iree/compiler/Dialect/HAL/Conversion:Passes",
    ],
)

cc_library(
    name = "init_compiler_modules",
    hdrs = ["init_compiler_modules.h"],
    deps = [
        "//iree/compiler/Dialect/Modules/Check/IR:CheckDialect",
    ],
)

cc_library(
    name = "init_targets",
    srcs = ["init_targets.cc"],
    hdrs = ["init_targets.h"],
    local_defines = [
        "IREE_HAVE_CUDA_TARGET",
        "IREE_HAVE_LLVMAOT_TARGET",
        "IREE_HAVE_METALSPIRV_TARGET",
        "IREE_HAVE_ROCM_TARGET",
        "IREE_HAVE_VMVX_TARGET",
        "IREE_HAVE_VULKANSPIRV_TARGET",
    ],
    deps = [
        "//iree/compiler/Dialect/HAL/Target/CUDA",
        "//iree/compiler/Dialect/HAL/Target/LLVM",
        "//iree/compiler/Dialect/HAL/Target/MetalSPIRV",
        "//iree/compiler/Dialect/HAL/Target/ROCM",
        "//iree/compiler/Dialect/HAL/Target/VMVX",
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
        "@mlir-hlo//:hlo",
        "@mlir-hlo//:lhlo",
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

cc_binary(
    name = "iree-opt",
    tags = ["hostonly"],
    deps = [
        ":iree_opt_main",
    ],
)

cc_library(
    name = "iree-mlir-lsp-server",
    srcs = ["iree-mlir-lsp-server.cc"],
    deps = [
        ":init_passes_and_dialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirLspServerLib",
        "@llvm-project//mlir:Support",
    ],
)

cc_binary(
    name = "iree-run-mlir",
    srcs = ["iree-run-mlir-main.cc"],
    tags = ["hostonly"],
    deps = [
        ":init_passes_and_dialects",
        ":init_targets",
        "//iree/base",
        "//iree/base:cc",
        "//iree/base:logging",
        "//iree/base:tracing",
        "//iree/base/internal:flags",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Dialect/VM/Target:init_targets",
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/compiler/Translation:IREEVM",
        "//iree/hal",
        "//iree/hal/drivers",
        "//iree/modules/hal",
        "//iree/tools/utils:vm_util",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)

cc_binary(
    name = "iree-run-module",
    srcs = ["iree-run-module-main.cc"],
    deps = [
        "//iree/base",
        "//iree/base:cc",
        "//iree/base:tracing",
        "//iree/base/internal:file_io",
        "//iree/base/internal:flags",
        "//iree/hal",
        "//iree/hal/drivers",
        "//iree/modules/hal",
        "//iree/tools/utils:vm_util",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:cc",
    ],
)

cc_binary(
    name = "iree-run-trace",
    srcs = ["iree-run-trace-main.c"],
    deps = [
        "//iree/base",
        "//iree/base:tracing",
        "//iree/base/internal:file_path",
        "//iree/base/internal:flags",
        "//iree/hal",
        "//iree/hal/drivers",
        "//iree/tools/utils:trace_replay",
        "//iree/tools/utils:yaml_util",
        "//iree/vm",
        "@com_github_yaml_libyaml//:yaml",
    ],
)

cc_binary(
    name = "iree-tblgen",
    srcs = [
        "//iree/compiler/Dialect/Util/Tools:GenSrcs",
        "//iree/compiler/Dialect/VM/Tools:GenSrcs",
    ],
    tags = ["hostonly"],
    deps = [
        "//build_tools:default_linkopts",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//mlir:MlirTableGenMain",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TableGen",
    ],
)

cc_library(
    name = "iree_translate_lib",
    srcs = ["iree_translate_lib.cc"],
    hdrs = ["iree_translate_lib.h"],
    deps = [
        ":init_compiler_modules",
        ":init_iree_passes_and_dialects",
        ":init_mlir_passes_and_dialects",
        ":init_passes_and_dialects",
        ":init_targets",
        ":init_translations",
        ":init_xla_dialects",
        "//iree/compiler/Codegen",
        "//iree/compiler/Dialect/VM/Target:init_targets",
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/compiler/Translation:IREEVM",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Translation",
    ],
)

cc_binary(
    name = "iree-translate",
    srcs = ["iree-translate-main.cc"],
    tags = ["hostonly"],
    deps = [
        ":iree_translate_lib",
    ],
)

sh_binary(
    name = "IreeFileCheck",
    testonly = True,
    srcs = ["IreeFileCheck.sh"],
    data = ["@llvm-project//llvm:FileCheck"],
    tags = ["hostonly"],
)
