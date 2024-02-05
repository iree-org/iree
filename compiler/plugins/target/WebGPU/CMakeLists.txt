# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tint is needed to compile SPIR-V into WGSL source code.
# Tint also requires SPIRV-Tools, which requires SPIRV-Headers.
#
# NOTE: these can be synced by referencing one of these repositories:
#   * https://dawn.googlesource.com/dawn/
#   * https://dawn.googlesource.com/tint/
#   * https://chromium.googlesource.com/vulkan-deps/+/refs/heads/main/DEPS
# or they can be updated independently
set(IREE_TINT_TAG          "fdb8787e9c1b79770bd98a8faf37fbe48a3077a4")  # 2023-03-06
set(IREE_SPIRV_HEADERS_TAG "5aa1dd8a11182ea9a6a0eabd6a9edc639d5dbecd")  # 2024-01-26
set(IREE_SPIRV_TOOLS_TAG   "80bc99c3d4a33fe07ae5dc93147bdd9495688b10")  # 2024-01-26
message(STATUS "Configuring WebGPU target deps (SPIRV-Headers, SPIRV-Tools, Tint)")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
add_subdirectory(spirv-headers EXCLUDE_FROM_ALL)
add_subdirectory(spirv-tools EXCLUDE_FROM_ALL)
add_subdirectory(tint EXCLUDE_FROM_ALL)
list(POP_BACK CMAKE_MESSAGE_INDENT)

add_subdirectory(test)

iree_compiler_register_plugin(
  PLUGIN_ID
    hal_target_webgpu
  TARGET
    ::WebGPU
)

iree_cc_library(
  NAME
    WebGPU
  HDRS
    "SPIRVToWGSL.h"
  SRCS
    "SPIRVToWGSL.cpp"
    "WebGPUTarget.cpp"
  DEPS
    LLVMSupport
    MLIRGPUDialect
    MLIRIR
    MLIRSPIRVDialect
    MLIRSPIRVSerialization
    MLIRSPIRVTransforms
    SPIRV-Tools
    iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
    iree::compiler::Codegen::SPIRV
    iree::compiler::Dialect::HAL::Target
    iree::compiler::PluginAPI
    iree::compiler::Utils
    iree::schemas::wgsl_executable_def_c_fbs
    libtint
  PUBLIC
)
