# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.13)

# CMake invokes the toolchain file twice during the first build, but only once
# during subsequent rebuilds. This was causing the various flags to be added
# twice on the first build, and on a rebuild ninja would see only one set of the
# flags and rebuild the world.
# https://github.com/android-ndk/ndk/issues/323
if(RISCV_TOOLCHAIN_INCLUDED)
  return()
endif()
set(RISCV_TOOLCHAIN_INCLUDED true)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT "${RISCV_TOOLCHAIN_ROOT}" STREQUAL "")
  set(CMAKE_AR           "${RISCV_TOOLCHAIN_ROOT}/bin/${RISCV_TOOLCHAIN_PREFIX}llvm-ar")
  set(CMAKE_C_COMPILER   "${RISCV_TOOLCHAIN_ROOT}/bin/${RISCV_TOOLCHAIN_PREFIX}clang")
  set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/${RISCV_TOOLCHAIN_PREFIX}clang++")
  set(CMAKE_RANLIB       "${RISCV_TOOLCHAIN_ROOT}/bin/${RISCV_TOOLCHAIN_PREFIX}llvm-ranlib")
  set(CMAKE_STRIP        "${RISCV_TOOLCHAIN_ROOT}/bin/${RISCV_TOOLCHAIN_PREFIX}llvm-strip")
  set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/riscv64-unknown-elf")
endif()

set(CMAKE_C_EXTENSIONS OFF CACHE BOOL "" FORCE) # runtime/src/iree/base/time.c:108:13: error: call to undeclared function 'clock_nanosleep'
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_THREADING OFF CACHE BOOL "" FORCE)
set(IREE_SYNCHRONIZATION_DISABLE_UNSAFE ON CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_TASK OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY OFF CACHE BOOL "" FORCE)

# Specify ISA spec for march=rv64gc. This is to resolve the mismatch between
# llvm and binutil ISA version.
set(RISCV_COMPILER_FLAGS "\
    -march=rv64i2p1ma2p1f2p2d2p2c2p0 -mabi=lp64d -DIREE_PLATFORM_GENERIC=1 \
    -DIREE_FILE_IO_ENABLE=0 -DIREE_TIME_NOW_FN=\"\{ return 0; \}\" -DIREE_DEVICE_SIZE_T=uint64_t -DPRIdsz=PRIu64")

set(CMAKE_C_FLAGS_INIT   "${RISCV_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${RISCV_COMPILER_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${RISCV_COMPILER_FLAGS}")

# GNUInstallDirs does not set CMAKE_INSTALL_LIBDIR in this configuration.
# When CMAKE_INSTALL_LIBDIR is unset, the install path becomes /cmake/IREE/IREERuntimeConfig.cmake
# When CMAKE_INSTALL_LIBDIR is set to "lib", the install path becomes ${CMAKE_INSTALL_PREFIX}/lib/cmake/IREE/IREERuntimeConfig.cmake
set(CMAKE_INSTALL_LIBDIR "lib" CACHE PATH "")
