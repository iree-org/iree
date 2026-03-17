# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)

# CMake invokes the toolchain file twice during the first build, but only once
# during subsequent rebuilds. This was causing the various flags to be added
# twice on the first build, and on a rebuild ninja would see only one set of the
# flags and rebuild the world.
# https://github.com/android-ndk/ndk/issues/323
if(IREE_WASM32_TOOLCHAIN_INCLUDED)
  return()
endif()
set(IREE_WASM32_TOOLCHAIN_INCLUDED true)

# Wasm32 is freestanding — no OS, no libc by default.
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR wasm32)

#-------------------------------------------------------------------------------
# wasi-sdk download
#-------------------------------------------------------------------------------

# Parse version and SHA-256 from the shared Bazel/CMake config file.
# This is the single source of truth — do not duplicate version constants here.
set(_WASI_SDK_VERSION_FILE "${CMAKE_CURRENT_LIST_DIR}/../wasm/wasi_sdk_version.bzl")
file(READ "${_WASI_SDK_VERSION_FILE}" _WASI_SDK_BZL)

string(REGEX MATCH "WASI_SDK_VERSION = \"([^\"]+)\"" _ "${_WASI_SDK_BZL}")
set(_WASI_SDK_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "WASI_SDK_TAG = \"([^\"]+)\"" _ "${_WASI_SDK_BZL}")
set(_WASI_SDK_TAG "${CMAKE_MATCH_1}")

# Detect host platform.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  set(_WASI_SDK_OS "linux")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  set(_WASI_SDK_OS "macos")
else()
  message(FATAL_ERROR "wasi-sdk: unsupported host OS: ${CMAKE_HOST_SYSTEM_NAME}")
endif()

cmake_host_system_information(RESULT _HOST_ARCH QUERY OS_PLATFORM)
if(_HOST_ARCH MATCHES "x86_64|AMD64")
  set(_WASI_SDK_ARCH "x86_64")
elseif(_HOST_ARCH MATCHES "aarch64|arm64")
  set(_WASI_SDK_ARCH "arm64")
else()
  message(FATAL_ERROR "wasi-sdk: unsupported host architecture: ${_HOST_ARCH}")
endif()

# Extract the SHA-256 hash for this platform from the .bzl dict.
set(_WASI_SDK_PLATFORM_KEY "${_WASI_SDK_ARCH}-${_WASI_SDK_OS}")
string(REGEX MATCH "\"${_WASI_SDK_PLATFORM_KEY}\": \"([^\"]+)\"" _ "${_WASI_SDK_BZL}")
set(_WASI_SDK_SHA256 "${CMAKE_MATCH_1}")
if("${_WASI_SDK_SHA256}" STREQUAL "")
  message(FATAL_ERROR "wasi-sdk: no SHA-256 hash for platform ${_WASI_SDK_PLATFORM_KEY}")
endif()

#-------------------------------------------------------------------------------
# Compiler tools
#-------------------------------------------------------------------------------

# Users can provide IREE_WASI_SDK_ROOT to skip the download.
# Otherwise we download wasi-sdk into the build directory.
if(NOT "${IREE_WASI_SDK_ROOT}" STREQUAL "")
  set(_WASI_SDK_ROOT "${IREE_WASI_SDK_ROOT}")
else()
  # Download wasi-sdk into the build tree (idempotent via stamp file).
  set(_WASI_SDK_BASENAME "wasi-sdk-${_WASI_SDK_VERSION}-${_WASI_SDK_ARCH}-${_WASI_SDK_OS}")
  set(_WASI_SDK_URL
    "https://github.com/WebAssembly/wasi-sdk/releases/download/${_WASI_SDK_TAG}/${_WASI_SDK_BASENAME}.tar.gz")

  # Download location: alongside the build directory so it persists across
  # reconfigures but is isolated per worktree.
  get_filename_component(_WASI_SDK_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/../wasi-sdk" ABSOLUTE)
  set(_WASI_SDK_ROOT "${_WASI_SDK_DOWNLOAD_DIR}/${_WASI_SDK_BASENAME}")
  set(_WASI_SDK_ARCHIVE "${_WASI_SDK_DOWNLOAD_DIR}/${_WASI_SDK_BASENAME}.tar.gz")
  set(_WASI_SDK_STAMP "${_WASI_SDK_DOWNLOAD_DIR}/${_WASI_SDK_BASENAME}.stamp")
  set(_WASI_SDK_STAMP_CONTENT "${_WASI_SDK_URL} : ${_WASI_SDK_SHA256}")

  set(_NEEDS_DOWNLOAD ON)
  if(EXISTS "${_WASI_SDK_STAMP}" AND IS_DIRECTORY "${_WASI_SDK_ROOT}")
    file(READ "${_WASI_SDK_STAMP}" _STAMP_CONTENTS)
    if("${_STAMP_CONTENTS}" STREQUAL "${_WASI_SDK_STAMP_CONTENT}")
      set(_NEEDS_DOWNLOAD OFF)
    endif()
  endif()

  if(_NEEDS_DOWNLOAD)
    message(STATUS "Downloading wasi-sdk ${_WASI_SDK_VERSION} for ${_WASI_SDK_PLATFORM_KEY} from ${_WASI_SDK_URL}")
    file(MAKE_DIRECTORY "${_WASI_SDK_DOWNLOAD_DIR}")
    file(DOWNLOAD "${_WASI_SDK_URL}" "${_WASI_SDK_ARCHIVE}"
      EXPECTED_HASH SHA256=${_WASI_SDK_SHA256}
      SHOW_PROGRESS)
    message(STATUS "Extracting wasi-sdk to ${_WASI_SDK_ROOT}")
    file(ARCHIVE_EXTRACT INPUT "${_WASI_SDK_ARCHIVE}" DESTINATION "${_WASI_SDK_DOWNLOAD_DIR}")
    file(REMOVE "${_WASI_SDK_ARCHIVE}")
    file(WRITE "${_WASI_SDK_STAMP}" "${_WASI_SDK_STAMP_CONTENT}")
  else()
    message(STATUS "Using cached wasi-sdk at ${_WASI_SDK_ROOT}")
  endif()
endif()

set(CMAKE_C_COMPILER   "${_WASI_SDK_ROOT}/bin/clang")
set(CMAKE_CXX_COMPILER "${_WASI_SDK_ROOT}/bin/clang++")
set(CMAKE_AR           "${_WASI_SDK_ROOT}/bin/llvm-ar")
set(CMAKE_RANLIB       "${_WASI_SDK_ROOT}/bin/llvm-ranlib")
set(CMAKE_STRIP        "${_WASI_SDK_ROOT}/bin/llvm-strip")
set(CMAKE_LINKER       "${_WASI_SDK_ROOT}/bin/wasm-ld")

# CMake's Generic platform module doesn't try test compiles by default,
# but we need to tell it our compiler works for the wasm32 target.
set(CMAKE_C_COMPILER_WORKS ON)
set(CMAKE_CXX_COMPILER_WORKS ON)

#-------------------------------------------------------------------------------
# Compiler and linker flags
#-------------------------------------------------------------------------------

# Wasm libc headers — shared between Bazel and CMake.
# Path is relative to the IREE source root.
set(IREE_WASM_LIBC_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/../wasm/libc/include")
get_filename_component(IREE_WASM_LIBC_INCLUDE "${IREE_WASM_LIBC_INCLUDE}" ABSOLUTE)

set(IREE_WASM32_COMPILE_FLAGS "\
  --target=wasm32-unknown-unknown \
  -nostdinc \
  -nostdlib \
  -ffreestanding \
  -isystem ${IREE_WASM_LIBC_INCLUDE} \
  -fno-exceptions \
  -fno-rtti \
  -fvisibility=hidden \
  -fno-short-wchar \
  -mbulk-memory \
  -msign-ext \
  -mnontrapping-fptoint \
  -matomics \
  -pthread \
  -DIREE_PLATFORM_WEB=1")

# Linker flags use -Wl, prefix because CMake invokes clang (not wasm-ld
# directly) as the linker driver. Clang strips the prefix and forwards
# to wasm-ld.
set(IREE_WASM32_LINK_FLAGS "\
  --target=wasm32-unknown-unknown \
  -nostdlib \
  -fuse-ld=lld \
  -Wl,--import-memory \
  -Wl,--shared-memory \
  -Wl,--no-entry \
  -Wl,--export-dynamic \
  -Wl,--allow-undefined \
  -Wl,--max-memory=4294967296")

set(CMAKE_C_FLAGS             "${IREE_WASM32_COMPILE_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS           "${IREE_WASM32_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_ASM_FLAGS           "${IREE_WASM32_COMPILE_FLAGS} ${CMAKE_ASM_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS    "${IREE_WASM32_LINK_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${IREE_WASM32_LINK_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${IREE_WASM32_LINK_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}")

#-------------------------------------------------------------------------------
# IREE build configuration
#-------------------------------------------------------------------------------

set(CMAKE_CROSSCOMPILING ON CACHE BOOL "")

# The compiler is not cross-compiled — it runs on the host. Either use
# pre-built host binaries (IREE_HOST_BIN_DIR) or an installed compiler.
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)

# Wasm has no filesystem, no dynamic library loading.
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE_JAVA OFF CACHE BOOL "" FORCE)

# HAL configuration: local drivers only (no GPU drivers in wasm).
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_TASK ON CACHE BOOL "" FORCE)

# Executable loaders: VMVX (interpreted) always available.
# Embedded ELF doesn't apply to wasm — wasm modules are the executable format.
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE ON CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS OFF CACHE BOOL "" FORCE)
