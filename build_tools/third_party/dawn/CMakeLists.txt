# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT (${IREE_HAL_WEBGPU_PLATFORM} STREQUAL "dawn"))
  return()
endif()

include(FetchContent)

FetchContent_Declare(
  dawn
  GIT_REPOSITORY https://dawn.googlesource.com/dawn
  GIT_TAG 908f4fe689b372439850e91858683a5273249cee # 2021-12-04
)

set(DAWN_ENABLE_OPENGL OFF)
set(DAWN_ENABLE_DESKTOP_GL OFF)
set(DAWN_ENABLE_OPENGLES OFF)
set(DAWN_USE_X11 OFF)
set(DAWN_BUILD_EXAMPLES OFF)

set(DAWN_THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party")
set(DAWN_SPIRV_CROSS_DIR "${DAWN_THIRD_PARTY_DIR}/spirv_cross")
set(DAWN_SPIRV_HEADERS_DIR "${DAWN_THIRD_PARTY_DIR}/spirv_headers")
set(DAWN_SPIRV_TOOLS_DIR "${DAWN_THIRD_PARTY_DIR}/spirv-tools")
set(DAWN_TINT_DIR "${DAWN_THIRD_PARTY_DIR}/tint")

# DO NOT SUBMIT iree_set_tint_cmake_options
macro(iree_set_tint_cmake_options)
set(TINT_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_TESTS OFF CACHE BOOL "" FORCE)

set(TINT_BUILD_SPV_READER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_READER OFF CACHE BOOL "" FORCE)

set(TINT_BUILD_GLSL_WRITER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_HLSL_WRITER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_MSL_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_SPV_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_WRITER ON CACHE BOOL "" FORCE)
endmacro()
iree_set_tint_cmake_options()

FetchContent_MakeAvailable(dawn)
FetchContent_GetProperties(dawn SOURCE_DIR DAWN_SOURCE)
