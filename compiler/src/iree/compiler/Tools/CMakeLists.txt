# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Doesn't use bazel_to_cmake because of various special logic throughout.

# Enable compiler targets based on options.
set(IREE_COMPILER_TARGETS "")
set(IREE_COMPILER_TARGET_COPTS "")
if(IREE_TARGET_BACKEND_DYLIB_LLVM_AOT OR IREE_TARGET_BACKEND_WASM_LLVM_AOT)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::LLVM)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_LLVMAOT_TARGET")
endif()
if(IREE_TARGET_BACKEND_METAL_SPIRV)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::MetalSPIRV)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_METALSPIRV_TARGET")
endif()
if(IREE_TARGET_BACKEND_VMVX)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::VMVX)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_VMVX_TARGET")
endif()
if(IREE_TARGET_BACKEND_VULKAN_SPIRV)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::VulkanSPIRV)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_VULKANSPIRV_TARGET")
endif()
if(IREE_TARGET_BACKEND_WEBGPU)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::WebGPU)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_WEBGPU_TARGET")
endif()
if(IREE_TARGET_BACKEND_CUDA)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::CUDA)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_CUDA_TARGET")
endif()
if(IREE_TARGET_BACKEND_ROCM)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::ROCM)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_ROCM_TARGET")
endif()

iree_cc_library(
  NAME
    init_iree_passes_and_dialects
  HDRS
    "init_iree_dialects.h"
    "init_iree_passes.h"
  DEPS
    IREEInputDialect
    IREELinalgExtDialect
    IREELinalgExtTransformOps
    IREELinalgExtTransforms
    IREELinalgExtOpInterfaceImpl
    IREELinalgTransformDialect
    IREELinalgTransformDialectTransforms
    IREEPyDMDialect
    MLIRIR
    iree::compiler::Bindings::Native::Transforms
    iree::compiler::Bindings::TFLite::Transforms
    iree::compiler::Codegen::Dialect::IREECodegenDialect
    iree::compiler::Codegen::Interfaces::Interfaces
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::Flow::Transforms
    iree::compiler::Dialect::HAL::IR::HALDialect
    iree::compiler::Dialect::HAL::Transforms
    iree::compiler::Dialect::Modules::VMVX::IR::VMVXDialect
    iree::compiler::Dialect::Modules::VMVX::Transforms
    iree::compiler::Dialect::Stream::IR
    iree::compiler::Dialect::Stream::Transforms
    iree::compiler::Dialect::Util::IR
    iree::compiler::Dialect::Util::Transforms
    iree::compiler::Dialect::VM::Analysis
    iree::compiler::Dialect::VM::IR
    iree::compiler::Dialect::VM::Transforms
    iree::compiler::Dialect::Vulkan::IR
    iree::compiler::InputConversion::Common
    iree::compiler::InputConversion::MHLO
    iree::compiler::InputConversion::TMTensor
    iree::compiler::InputConversion::TOSA
    iree::compiler::ConstEval
    iree::compiler::Translation::HALExecutable
    iree::compiler::Translation::IREEVM
  PUBLIC
)

iree_cc_library(
  NAME
    init_mlir_passes_and_dialects
  HDRS
    "init_mlir_dialects.h"
    "init_mlir_passes.h"
  DEPS
    # Sets IREE_HAVE_EMITC_DIALECT and transitively depends on MLIREmitC
    # if enabled.
    iree::compiler::Dialect::VM::Target::C::Enabled
    MLIRAffine
    MLIRAffineTransforms
    MLIRArmNeon
    MLIRArmNeon2dToIntr
    MLIRBufferization
    MLIRControlFlow
    MLIRGPUOps
    MLIRGPUToSPIRV
    MLIRIR
    MLIRLLVMIR
    MLIRLinalg
    MLIRLinalgToLLVM
    MLIRLinalgToSPIRV
    MLIRLinalgTransforms
    MLIRQuant
    MLIRQuantTransforms
    MLIRSCF
    MLIRSCFToGPU
    MLIRSCFTransforms
    MLIRSPIRV
    MLIRSPIRVTransforms
    MLIRShape
    MLIRFunc
    MLIRFuncToSPIRV
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRTosa
    MLIRTosaTransforms
    MLIRTransforms
    MLIRVector
  PUBLIC
)

if(IREE_BUILD_TORCH_MLIR_SUPPORT)
  set(_OPTIONAL_INIT_TORCH_MLIR_DIALECT_LIB ::init_torch_mlir_dialects)
endif()

iree_cc_library(
  NAME
    init_passes_and_dialects
  HDRS
    "init_dialects.h"
    "init_passes.h"
  DEPS
    ::init_compiler_modules
    ::init_iree_passes_and_dialects
    ::init_mlir_passes_and_dialects
    ${_OPTIONAL_INIT_TORCH_MLIR_DIALECT_LIB}
    ::init_xla_dialects
    iree::compiler::Codegen::Codegen
    iree::compiler::Dialect::HAL::Conversion::Passes
  PUBLIC
)

iree_cc_library(
  NAME
    init_llvmir_translations
  HDRS
    "init_llvmir_translations.h"
  DEPS
    MLIRArmNeonToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
  PUBLIC
)

iree_cc_library(
  NAME
    init_compiler_modules
  HDRS
    "init_compiler_modules.h"
  DEPS
    iree::compiler::Dialect::Modules::Check::IR::CheckDialect
)

iree_cc_library(
  NAME
    init_targets
  HDRS
    "init_targets.h"
  SRCS
    "init_targets.cc"
  DEPS
    ${IREE_COMPILER_TARGETS}
  COPTS
    ${IREE_COMPILER_TARGET_COPTS}
  PUBLIC
)

iree_cc_library(
  NAME
    init_xla_dialects
  HDRS
    "init_xla_dialects.h"
  DEPS
    tensorflow::external_mhlo_includes
    ChloDialect
    MhloDialect
  PUBLIC
)

if(IREE_BUILD_TORCH_MLIR_SUPPORT)
  iree_cc_library(
    NAME
      init_torch_mlir_dialects
    HDRS
      "init_torch_mlir_dialects.h"
    DEPS
      TorchMLIRTMTensorDialect
    DEFINES
      "IREE_HAVE_TORCH_MLIR_DIALECTS"
    PUBLIC
  )
endif()

iree_cc_library(
  NAME
    init_translations
  HDRS
    "init_translations.h"
  DEPS
    iree::compiler::Translation::HALExecutable
    iree::compiler::Translation::IREEVM
    MLIRSPIRVTranslateRegistration
)

iree_cc_library(
  NAME
    iree_translate_lib
  HDRS
    "iree_translate_lib.h"
  SRCS
    "iree_compile_lib.cc"
    "iree_translate_lib.cc"
  DEPS
    ::init_llvmir_translations
    ::init_passes_and_dialects
    ::init_targets
    ::init_translations
    LLVMSupport
    MLIRIR
    MLIRSCFTransforms
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTranslateLib
    iree::compiler::Codegen::Codegen
    iree::compiler::ConstEval
    iree::compiler::Dialect::VM::Target::Bytecode
    iree::compiler::Dialect::VM::Target::init_targets
    iree::compiler::Dialect::VM::Target::C
    iree::compiler::Pipelines
    iree::compiler::Translation::HALExecutable
    iree::compiler::Translation::IREEVM
    iree::compiler::Utils
  PUBLIC
)
