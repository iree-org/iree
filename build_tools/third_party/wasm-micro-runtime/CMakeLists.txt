# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file modeled after the instructions at
# https://github.com/bytecodealliance/wasm-micro-runtime/blob/main/doc/build_wamr.md

set(WAMR_SOURCE_DIR "${IREE_SOURCE_DIR}/third_party/wasm-micro-runtime")

# Runtime build target -> platform in wasm-micro-runtime/core/shared/platform/
if(ANDROID)
  set(WAMR_BUILD_PLATFORM "android")
elseif(APPLE)
  set(WAMR_BUILD_PLATFORM "darwin")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(WAMR_BUILD_PLATFORM "linux")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(WAMR_BUILD_PLATFORM "windows")
  # Enable MASM (Microsoft Macro Assembler) for compiling .asm files.
  enable_language(ASM_MASM)
endif()

# Pick the features we want. Most default to disabled.
set(WAMR_BUILD_INTERP 1)

include(${WAMR_SOURCE_DIR}/build-scripts/runtime_lib.cmake)

# WAMR unconditionally adds a global compiler flag for 64 bit targets, which
# is not handled or needed by MSVC.
# TODO(scotttodd): Omit the flag upstream rather than work around it here.
if(MSVC)
  string(REGEX REPLACE " -fPIC" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif()

set(_WAMR_LIBRARY_NAME "wasm_micro_runtime_vmlib")
add_library(${_WAMR_LIBRARY_NAME} STATIC "")
target_sources(${_WAMR_LIBRARY_NAME}
  PRIVATE
    ${WAMR_RUNTIME_LIB_SOURCE})
target_include_directories(${_WAMR_LIBRARY_NAME} SYSTEM
  PUBLIC
    "$<BUILD_INTERFACE:${WAMR_SOURCE_DIR}/core/iwasm/include/>")

# Suppress all warnings for this third party code.
if(MSVC)
  target_compile_options(${_WAMR_LIBRARY_NAME} PRIVATE "/W0")
else()
  target_compile_options(${_WAMR_LIBRARY_NAME} PRIVATE "-w")
endif()
