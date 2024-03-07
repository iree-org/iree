# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Doesn't use bazel_to_cmake because of various special logic throughout.

# Enable compiler targets based on options.
set(IREE_COMPILER_TARGETS "")
set(IREE_COMPILER_TARGET_COPTS "")
if(IREE_TARGET_BACKEND_LLVM_CPU)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::LLVMCPU)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_LLVM_CPU_TARGET")
endif()

set(IREE_VERSION_TARGET_COPTS "")
if(IREE_EMBEDDED_RELEASE_INFO)
  list(APPEND IREE_VERSION_TARGET_COPTS "-DIREE_RELEASE_REVISION=\"${IREE_RELEASE_REVISION}\"")
  list(APPEND IREE_VERSION_TARGET_COPTS "-DIREE_RELEASE_VERSION=\"${IREE_RELEASE_VERSION}\"")
endif()

iree_cc_library(
  NAME
    init_iree_passes_and_dialects
  HDRS
    "init_iree_dialects.h"
    "init_iree_passes.h"
  DEPS
    IREEInputDialect
    IREELinalgTransformDialect
    IREELinalgTransformDialectPasses
    IREEVectorExtDialect
    MLIRIR
    iree::compiler::Bindings::Native::Transforms
    iree::compiler::Bindings::TFLite::Transforms
    iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
    iree::compiler::Codegen::Dialect::GPU::IR::IREEGPUDialect
    iree::compiler::Codegen::Interfaces::Interfaces
    iree::compiler::ConstEval
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::Flow::Transforms
    iree::compiler::Dialect::HAL::IR::HALDialect
    iree::compiler::Dialect::HAL::Transforms
    iree::compiler::Dialect::LinalgExt::IR
    iree::compiler::Dialect::LinalgExt::TransformExtensions::LinalgExtExtensions
    iree::compiler::Dialect::LinalgExt::Transforms
    iree::compiler::Dialect::Stream::IR
    iree::compiler::Dialect::Stream::Transforms
    iree::compiler::Dialect::Util::IR
    iree::compiler::Dialect::Util::TransformOps
    iree::compiler::Dialect::Util::Transforms
    iree::compiler::Dialect::VM::Analysis
    iree::compiler::Dialect::VM::IR
    iree::compiler::Dialect::VM::Target::init_targets
    iree::compiler::Dialect::VM::Transforms
    iree::compiler::Dialect::VMVX::IR::VMVXDialect
    iree::compiler::Dialect::VMVX::Transforms
    iree::compiler::Dialect::Vulkan::IR
    iree::compiler::ExternalInterfaces::ExternalModels
    iree::compiler::GlobalOptimization::Interfaces::Interfaces
    iree::compiler::InputConversion::Common
    iree::compiler::Modules::HAL::Inline::IR::HALInlineDialect
    iree::compiler::Modules::HAL::Inline::Transforms
    iree::compiler::Modules::HAL::Loader::IR::HALLoaderDialect
    iree::compiler::Modules::HAL::Loader::Transforms
    iree::compiler::Modules::IO::Parameters::IR::IOParametersDialect
    iree::compiler::Modules::IO::Parameters::Transforms
    iree::compiler::Pipelines
    iree::compiler::Preprocessing::Passes
    iree::compiler::Preprocessing::TransformExtensions::PreprocessingExtensions
  PUBLIC
)

iree_cc_library(
  NAME
    init_mlir_passes_and_dialects
  HDRS
    "init_mlir_dialects.h"
    "init_mlir_passes.h"
  DEPS
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRArmNeonDialect
    MLIRArmNeon2dToIntr
    MLIRBufferizationDialect
    MLIRComplexDialect
    MLIRControlFlowDialect
    MLIRFuncInlinerExtension
    MLIRGPUDialect
    MLIRGPUToSPIRV
    MLIRIR
    MLIRLLVMDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMLProgramDialect
    MLIRQuantDialect
    MLIRSCFDialect
    MLIRSCFToGPU
    MLIRSCFTransforms
    MLIRSPIRVDialect
    MLIRSPIRVTransforms
    MLIRShapeDialect
    MLIRFuncDialect
    MLIRFuncToSPIRV
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRTransformDialect
    MLIRTransforms
    MLIRVectorDialect
    iree::compiler::Dialect::VM::Target::init_targets

    MLIRAffineTransformOps
    MLIRBufferizationTransformOps
    MLIRFuncTransformOps
    MLIRGPUTransformOps
    MLIRLinalgTransformOps
    MLIRMemRefTransformOps
    MLIRSCFTransformOps
    MLIRTensorTransformOps
    MLIRVectorTransformOps
    MLIRTransformLoopExtension
  PUBLIC
)

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
    iree::compiler::Modules::Check::IR::CheckDialect
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
    iree_compile_lib
  HDRS
    "iree_compile_lib.h"
  SRCS
    "iree_compile_lib.cc"
  DEPS
    LLVMSupport
    iree::compiler::API::Internal::CompilerDriver
    iree::compiler::Pipelines
    iree::compiler::bindings::c::headers
  PUBLIC
)

iree_cc_library(
  NAME
    version
  HDRS
    "version.h"
  SRCS
    "version.cc"
  COPTS
    ${IREE_VERSION_TARGET_COPTS}
)
