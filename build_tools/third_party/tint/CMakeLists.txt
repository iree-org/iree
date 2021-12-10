# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT "${IREE_TARGET_BACKEND_WEBGPU}" OR
   NOT (${IREE_HAL_WEBGPU_PLATFORM} STREQUAL "dawn"))
  return()
endif()

include(FetchContent)

FetchContent_Declare(
  tint
  GIT_REPOSITORY https://dawn.googlesource.com/tint
  GIT_TAG 4e6e1138160a7a003c5d09e6d6e9ba45dcb50347 # 2021-12-10
)

set(TINT_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_TESTS OFF CACHE BOOL "" FORCE)

set(TINT_BUILD_SPV_READER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_READER ON CACHE BOOL "" FORCE)

set(TINT_BUILD_GLSL_WRITER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_HLSL_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_MSL_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_SPV_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_WRITER ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(tint)
FetchContent_GetProperties(tint SOURCE_DIR TINT_SOURCE)
