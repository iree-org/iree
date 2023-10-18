# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_VULKAN_SPIRV OR NOT IREE_HAL_DRIVER_VULKAN)
  return()
endif()

# NOTE: we use glslc (either provided by user or in the Vulkan SDK) to do our
# .glsl -> .spv compilation. This example is just demonstrating how to use
# custom shaders in .spv format and not supporting infrastructure for compiling
# shaders from various textual input languages (HLSL/etc). Users are expected to
# bring their own infrastructure if they want to bring their own source code.
find_program(GLSLC glslc HINTS "D:\\Tools\\VulkanSDK\\1.3.261.1\\Bin\\glslc.exe")
if(NOT GLSLC)
  message(STATUS "IREE custom_dispatch/vulkan/shaders ignored -- glslc not found")
  return()
endif()

set(_SPV_TARGET iree_samples_custom_dispatch_vulkan_shaders_spv)
add_custom_command(
  OUTPUT simple_mul.spv
  DEPENDS simple_mul.glsl
  COMMAND ${GLSLC}
      -fshader-stage=compute
      -o simple_mul.spv
      ${CMAKE_CURRENT_SOURCE_DIR}/simple_mul.glsl
  VERBATIM
)
add_custom_command(
  OUTPUT simple_mul_inplace.spv
  DEPENDS simple_mul_inplace.glsl
  COMMAND ${GLSLC}
      -fshader-stage=compute
      -o simple_mul_inplace.spv
      ${CMAKE_CURRENT_SOURCE_DIR}/simple_mul_inplace.glsl
  VERBATIM
)
add_custom_target(iree_samples_custom_dispatch_vulkan_shaders_spv DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.spv
  ${CMAKE_CURRENT_BINARY_DIR}/simple_mul_inplace.spv
)
add_dependencies(iree-sample-deps "${_SPV_TARGET}")

iree_lit_test_suite(
  NAME
    example
  SRCS
    "example.mlir"
    "example_inline.mlir"
    "example_pattern_module.mlir"
  DATA
    ${_SPV_TARGET}
    "example_patterns.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  LABELS
    "driver=vulkan"
    "hostonly"
)
