# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_WEBGPU)
  return()
endif()

iree_add_all_subdirs()

iree_cc_library(
  NAME
    WebGPU
  HDRS
    "SPIRVToWGSL.h"
    "WebGPUTarget.h"
  SRCS
    "SPIRVToWGSL.cpp"
    "WebGPUTarget.cpp"
  DEPS
    LLVMSupport
    MLIRGPUOps
    MLIRIR
    MLIRSPIRVDialect
    MLIRSPIRVSerialization
    MLIRSPIRVTransforms
    SPIRV-Tools
    iree::compiler::Codegen::Dialect::IREECodegenDialect
    iree::compiler::Codegen::SPIRV
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Utils
    iree::schemas::wgsl_executable_def_c_fbs
    libtint
  PUBLIC
)
