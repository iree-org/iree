# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "init_iree_passes_and_dialects",
    hdrs = [
        "init_iree_dialects.h",
        "init_iree_passes.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Bindings/Native/Transforms",
        "//compiler/src/iree/compiler/Bindings/TFLite/Transforms",
        "//compiler/src/iree/compiler/Codegen/Dialect:IREECodegenDialect",
        "//compiler/src/iree/compiler/Codegen/Interfaces",
        "//compiler/src/iree/compiler/ConstEval",
        "//compiler/src/iree/compiler/Dialect/Flow/IR",
        "//compiler/src/iree/compiler/Dialect/Flow/Transforms",
        "//compiler/src/iree/compiler/Dialect/HAL/IR:HALDialect",
        "//compiler/src/iree/compiler/Dialect/HAL/Transforms",
        "//compiler/src/iree/compiler/Dialect/Modules/VMVX/IR:VMVXDialect",
        "//compiler/src/iree/compiler/Dialect/Modules/VMVX/Transforms",
        "//compiler/src/iree/compiler/Dialect/Stream/IR",
        "//compiler/src/iree/compiler/Dialect/Stream/Transforms",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Dialect/Util/Transforms",
        "//compiler/src/iree/compiler/Dialect/VM/Analysis",
        "//compiler/src/iree/compiler/Dialect/VM/IR",
        "//compiler/src/iree/compiler/Dialect/VM/Transforms",
        "//compiler/src/iree/compiler/Dialect/Vulkan/IR",
        "//compiler/src/iree/compiler/InputConversion/Common",
        "//compiler/src/iree/compiler/InputConversion/MHLO",
        "//compiler/src/iree/compiler/InputConversion/TMTensor",
        "//compiler/src/iree/compiler/InputConversion/TOSA",
        "//compiler/src/iree/compiler/Translation:HALExecutable",
        "//compiler/src/iree/compiler/Translation:IREEVM",
        "//llvm-external-projects/iree-dialects:IREEInputDialect",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "//llvm-external-projects/iree-dialects:IREELinalgExtPasses",
        "//llvm-external-projects/iree-dialects:IREELinalgTransformDialect",
        "//llvm-external-projects/iree-dialects:IREELinalgTransformDialectPasses",
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
        "//compiler/src/iree/compiler/Dialect/VM/Target/C:Enabled",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:ArmNeon",
        "@llvm-project//mlir:ArmNeon2dToIntr",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ControlFlowOps",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:EmitC",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToSPIRV",
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
        "@llvm-project//mlir:TensorInferTypeOpInterfaceImpl",
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
        ":init_torch_mlir_dialects",
        ":init_xla_dialects",
        "//compiler/src/iree/compiler/Codegen",
        "//compiler/src/iree/compiler/Dialect/HAL/Conversion:Passes",
    ],
)

cc_library(
    name = "init_compiler_modules",
    hdrs = ["init_compiler_modules.h"],
    deps = [
        "//compiler/src/iree/compiler/Dialect/Modules/Check/IR:CheckDialect",
    ],
)

cc_library(
    name = "init_targets",
    srcs = ["init_targets.cc"],
    hdrs = ["init_targets.h"],
    local_defines = [
        # "IREE_HAVE_CUDA_TARGET",
        "IREE_HAVE_LLVMAOT_TARGET",
        "IREE_HAVE_METALSPIRV_TARGET",
        "IREE_HAVE_ROCM_TARGET",
        "IREE_HAVE_VMVX_TARGET",
        "IREE_HAVE_VULKANSPIRV_TARGET",
    ],
    deps = [
        # "//compiler/src/iree/compiler/Dialect/HAL/Target/CUDA",
        "//compiler/src/iree/compiler/Dialect/HAL/Target/LLVM",
        "//compiler/src/iree/compiler/Dialect/HAL/Target/MetalSPIRV",
        "//compiler/src/iree/compiler/Dialect/HAL/Target/ROCM",
        "//compiler/src/iree/compiler/Dialect/HAL/Target/VMVX",
        "//compiler/src/iree/compiler/Dialect/HAL/Target/VulkanSPIRV",
    ],
)

cc_library(
    name = "init_translations",
    hdrs = [
        "init_translations.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Translation:HALExecutable",
        "//compiler/src/iree/compiler/Translation:IREEVM",
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
    name = "init_torch_mlir_dialects",
    hdrs = [
        "init_torch_mlir_dialects.h",
    ],
    deps = [
        "@torch-mlir-dialects//:TorchMLIRTMTensorDialect",
    ],
)

cc_library(
    name = "init_llvmir_translations",
    hdrs = [
        "init_llvmir_translations.h",
    ],
    deps = [
        "@llvm-project//mlir:ArmNeonToLLVMIRTranslation",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)

cc_library(
    name = "iree_translate_lib",
    srcs = [
        "iree_compile_lib.cc",
        "iree_translate_lib.cc",
    ],
    hdrs = ["iree_translate_lib.h"],
    deps = [
        ":init_llvmir_translations",
        ":init_passes_and_dialects",
        ":init_targets",
        ":init_translations",
        "//compiler/src/iree/compiler/Codegen",
        "//compiler/src/iree/compiler/ConstEval",
        "//compiler/src/iree/compiler/Dialect/VM/Target:init_targets",
        "//compiler/src/iree/compiler/Dialect/VM/Target/Bytecode",
        "//compiler/src/iree/compiler/Dialect/VM/Target/C",
        "//compiler/src/iree/compiler/Pipelines",
        "//compiler/src/iree/compiler/Translation:HALExecutable",
        "//compiler/src/iree/compiler/Translation:IREEVM",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)
