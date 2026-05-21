# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Dawn's Tint libraries are needed to compile SPIR-V into WGSL source code.
# Tint also requires Abseil and SPIRV-Tools, which requires SPIRV-Headers.
#
# NOTE: these can be synced by referencing one of these repositories:
#   * https://dawn.googlesource.com/dawn/
#   * https://chromium.googlesource.com/vulkan-deps/+/refs/heads/main/DEPS
# or they can be updated independently
set(IREE_DAWN_TAG          "851ba3e50c354ef66d16c518d4341c01ed6828cc")
set(IREE_ABSEIL_TAG        "04f3bc01d12cf58c90a1bb68990f087fa3c3ed19")
set(IREE_SPIRV_HEADERS_TAG "465055f6c9128772e20082e893d974146acf7a02")
set(IREE_SPIRV_TOOLS_TAG   "5a1eea1546c372a945a27d9b10e0a059db6cc651")
message(STATUS "Configuring WebGPU target deps (SPIRV-Headers, SPIRV-Tools, Abseil, Dawn/Tint)")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
add_subdirectory(spirv-headers EXCLUDE_FROM_ALL)
add_subdirectory(spirv-tools EXCLUDE_FROM_ALL)
add_subdirectory(dawn EXCLUDE_FROM_ALL)
list(POP_BACK CMAKE_MESSAGE_INDENT)

add_subdirectory(test)

iree_compiler_register_plugin(
  PLUGIN_ID
    hal_target_webgpu_spirv
  TARGET
    ::WebGPUSPIRV
)

iree_cc_library(
  NAME
    SPIRVToWGSL
  HDRS
    "SPIRVToWGSL.h"
  SRCS
    "SPIRVToWGSL.cpp"
  DEPS
    LLVMSupport
    SPIRV-Tools
    libtint
  PUBLIC
)

# Tint's public headers use C++20 features such as concepts and std::span.
iree_package_name(_WEBGPU_SPIRV_PACKAGE_NAME)
set_property(
  TARGET ${_WEBGPU_SPIRV_PACKAGE_NAME}_SPIRVToWGSL.objects
  PROPERTY CXX_STANDARD 20
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(${_WEBGPU_SPIRV_PACKAGE_NAME}_SPIRVToWGSL.objects
    PRIVATE
      -Wno-unknown-warning-option
  )
endif()

iree_cc_library(
  NAME
    WebGPUSPIRV
  SRCS
    "WebGPUSPIRVTarget.cpp"
  DEPS
    ::SPIRVToWGSL
    LLVMSupport
    MLIRGPUDialect
    MLIRIR
    MLIRSPIRVDialect
    MLIRSPIRVSerialization
    MLIRSPIRVTransforms
    iree::compiler::Codegen::Common::PassHeaders
    iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
    iree::compiler::Codegen::Dialect::GPU::TargetUtils::KnownTargets
    iree::compiler::Codegen::SPIRV
    iree::compiler::Codegen::WGSL
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Dialect::HAL::Utils::ExecutableDebugInfoUtils
    iree::compiler::PluginAPI
    iree::compiler::Utils
    iree::schemas::executable_debug_info_c_fbs
    iree::schemas::webgpu_executable_def_c_fbs
  PUBLIC
)
