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
if(IREE_WASM32_WASI_TOOLCHAIN_INCLUDED)
  return()
endif()
set(IREE_WASM32_WASI_TOOLCHAIN_INCLUDED true)

# WASI is a hosted environment — musl libc, libc++, full C/C++ stdlib.
# For freestanding production builds, use wasm32.cmake instead.
set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_PROCESSOR wasm32)

#-------------------------------------------------------------------------------
# wasi-sdk download (shared with wasm32.cmake)
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
set(CMAKE_SYSROOT      "${_WASI_SDK_ROOT}/share/wasi-sysroot")

# Tell CMake we know the compiler works (skip test compile for cross toolchain).
set(CMAKE_C_COMPILER_WORKS ON)
set(CMAKE_CXX_COMPILER_WORKS ON)

#-------------------------------------------------------------------------------
# Compiler and linker flags
#-------------------------------------------------------------------------------

# Detect the clang resource directory version for explicit -resource-dir.
file(GLOB _CLANG_VERSION_DIRS "${_WASI_SDK_ROOT}/lib/clang/*")
list(LENGTH _CLANG_VERSION_DIRS _CLANG_VERSION_COUNT)
if(NOT _CLANG_VERSION_COUNT EQUAL 1)
  message(FATAL_ERROR "Expected exactly one clang version directory under lib/clang/, got: ${_CLANG_VERSION_DIRS}")
endif()
list(GET _CLANG_VERSION_DIRS 0 _CLANG_RESOURCE_DIR)

set(IREE_WASM32_WASI_COMPILE_FLAGS "\
  --target=wasm32-wasi \
  --sysroot=${CMAKE_SYSROOT} \
  -resource-dir=${_CLANG_RESOURCE_DIR} \
  -fno-exceptions \
  -fno-rtti \
  -fvisibility=hidden \
  -fno-short-wchar \
  -mbulk-memory \
  -msign-ext \
  -mnontrapping-fptoint \
  -DIREE_PLATFORM_WEB=1 \
  -DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1 \
  -DIREE_THREADING_ENABLE=0 \
  -D_WASI_EMULATED_SIGNAL \
  -D_WASI_EMULATED_PROCESS_CLOCKS \
  -DCLOCK_THREAD_CPUTIME_ID=CLOCK_MONOTONIC \
  -DBENCHMARK_OS_NACL \
  -DGTEST_HAS_STREAM_REDIRECTION=0 \
  -DGTEST_HAS_EXCEPTIONS=0")

# C++ flags: explicit include paths for libc++ headers.
# clang's auto-detection of C++ include paths can fail in non-standard
# directory layouts (symlinks, relocated SDKs). Explicit -isystem
# makes the build independent of clang's InstalledDir probing.
set(IREE_WASM32_WASI_CXX_FLAGS "\
  -std=c++17 \
  -isystem ${CMAKE_SYSROOT}/include/wasm32-wasi/c++/v1 \
  -isystem ${CMAKE_SYSROOT}/include/c++/v1")

set(IREE_WASM32_WASI_LINK_FLAGS "\
  --target=wasm32-wasi \
  --sysroot=${CMAKE_SYSROOT} \
  -resource-dir=${_CLANG_RESOURCE_DIR} \
  -fuse-ld=lld \
  -Wl,--undefined=__main_argc_argv \
  -lwasi-emulated-signal \
  -lwasi-emulated-process-clocks \
  -lc++ \
  -lc++abi")

set(CMAKE_C_FLAGS             "${IREE_WASM32_WASI_COMPILE_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS           "${IREE_WASM32_WASI_COMPILE_FLAGS} ${IREE_WASM32_WASI_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_ASM_FLAGS           "${IREE_WASM32_WASI_COMPILE_FLAGS} ${CMAKE_ASM_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS    "${IREE_WASM32_WASI_LINK_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${IREE_WASM32_WASI_LINK_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS "${IREE_WASM32_WASI_LINK_FLAGS}" CACHE STRING "" FORCE)

#-------------------------------------------------------------------------------
# IREE build configuration
#-------------------------------------------------------------------------------

set(CMAKE_CROSSCOMPILING ON CACHE BOOL "")

# Disable gtest features unavailable on WASI. gtest_disable_pthreads prevents
# gtest from adding -lpthread. GTEST_HAS_EXCEPTIONS is also set in CFLAGS
# above, but gtest's CMakeLists.txt adds -fexceptions explicitly unless we
# tell it not to via this cache variable.
set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)

# Wasm test binaries are not directly executable on the host. Tests that need
# to run under wasm use their own runner (e.g., iree_wasm_cc_test with the
# bundled Node.js entry point). No CMAKE_CROSSCOMPILING_EMULATOR is set.
find_program(NODE_EXECUTABLE node REQUIRED)

# The compiler is not cross-compiled — it runs on the host.
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)

set(IREE_BUILD_TESTS ON CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE_JAVA OFF CACHE BOOL "" FORCE)

# WASI preview1 is single-threaded. Disable synchronization and threading
# so iree_copts.cmake injects the correct -D flags for all IREE targets.
# The -D flags in IREE_WASM32_WASI_COMPILE_FLAGS above handle non-IREE
# targets (gtest, benchmark) that don't use iree_copts.
set(IREE_ENABLE_THREADING OFF CACHE BOOL "" FORCE)
set(IREE_SYNCHRONIZATION_DISABLE_UNSAFE ON CACHE BOOL "" FORCE)

# HAL configuration: local-sync only (local-task requires threading).
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_TASK OFF CACHE BOOL "" FORCE)

# Executable loaders: VMVX (interpreted) always available.
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE ON CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS OFF CACHE BOOL "" FORCE)
