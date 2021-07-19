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
  abseil-cpp
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG ${IREE_DAWN_ABSEIL_TAG}
)

set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(abseil-cpp)
FetchContent_GetProperties(abseil-cpp SOURCE_DIR ABSEIL_CPP_SOURCE)

# FetchContent_Declare(
#   glfw
#   GIT_REPOSITORY https://github.com/glfw/glfw.git
#   GIT_TAG ${IREE_DAWN_GLFW_TAG}
# )
# FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
  dawn
  GIT_REPOSITORY https://dawn.googlesource.com/dawn
  GIT_TAG ${IREE_DAWN_TAG}
)

set(DAWN_ENABLE_OPENGL OFF)
set(DAWN_ENABLE_DESKTOP_GL OFF)
set(DAWN_ENABLE_OPENGLES OFF)
set(DAWN_USE_X11 OFF)
set(DAWN_SUPPORTS_GLFW_FOR_WINDOWING OFF)
set(DAWN_BUILD_EXAMPLES OFF)

set(DAWN_THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party")
set(DAWN_ABSEIL_DIR "${ABSEIL_CPP_SOURCE}")
set(DAWN_SPIRV_CROSS_DIR "${DAWN_THIRD_PARTY_DIR}/spirv_cross")
set(DAWN_SPIRV_HEADERS_DIR "${DAWN_THIRD_PARTY_DIR}/spirv_headers")
set(DAWN_SPIRV_TOOLS_DIR "${SPIRV_TOOLS_SOURCE}")
set(DAWN_TINT_DIR "${TINT_SOURCE}")

# HACK: dawn looks for these targets instead of trusting the dirs set above,
# leading to weird double inclusion of things like abseil. I have no idea why
# it requires glfw even when opengl and examples are off.
add_library(libabsl STATIC EXCLUDE_FROM_ALL dummy.c)

FetchContent_MakeAvailable(dawn)
FetchContent_GetProperties(dawn SOURCE_DIR DAWN_SOURCE)
