# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)

FetchContent_Declare(
  tint
  GIT_REPOSITORY https://dawn.googlesource.com/tint
  GIT_TAG 188b1fb8f5be52299fb7fbc6db17dbb0c07dbb7e # 2021-12-16
)

set(TINT_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# Our usage at compile time primarily needs the SPIR-V reader and WGSL writer,
# but usage at runtime through Dawn needs a broader set.

set(TINT_BUILD_SPV_READER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_READER ON CACHE BOOL "" FORCE)

set(TINT_BUILD_GLSL_WRITER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_HLSL_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_MSL_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_SPV_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_WRITER ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(tint)
FetchContent_GetProperties(tint SOURCE_DIR TINT_SOURCE)
