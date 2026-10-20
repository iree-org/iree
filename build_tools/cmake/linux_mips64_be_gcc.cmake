# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Big-endian MIPS64 (mips64, n64 ABI) toolchain [SPIKE].
# Same MIPS64R2 ISA and n64 data model as linux_mips64_gcc.cmake, but
# BIG-ENDIAN (mips64, not mips64el). Produces ELF 64-bit MSB binaries, run
# under qemu-mips64 (big-endian). IREE_ARCH is still "mips_64"; endianness is
# detected from the compiler (__BYTE_ORDER__) in base/target_platform.h.

cmake_minimum_required(VERSION 3.26)

if(MIPS_TOOLCHAIN_INCLUDED)
  return()
endif()
set(MIPS_TOOLCHAIN_INCLUDED true)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR mips64)

set(MIPS_TOOLCHAIN_PREFIX "${MIPS_TOOLCHAIN_PREFIX}")
if(MIPS_TOOLCHAIN_PREFIX STREQUAL "")
  set(MIPS_TOOLCHAIN_PREFIX "mips64-linux-gnuabi64-")
endif()

if(NOT "${MIPS_TOOLCHAIN_ROOT}" STREQUAL "")
  set(CMAKE_AR           "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}ar")
  set(CMAKE_C_COMPILER   "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}gcc")
  set(CMAKE_CXX_COMPILER "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}g++")
  set(CMAKE_RANLIB       "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}ranlib")
  set(CMAKE_STRIP        "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}strip")
  set(CMAKE_SYSROOT      "${MIPS_TOOLCHAIN_ROOT}/sysroot")
  set(MIPS_SYSROOT       "${MIPS_TOOLCHAIN_ROOT}/sysroot")
else()
  set(CMAKE_C_COMPILER   "${MIPS_TOOLCHAIN_PREFIX}gcc")
  set(CMAKE_CXX_COMPILER "${MIPS_TOOLCHAIN_PREFIX}g++")
  # Distro big-endian cross root; only used for QEMU's -L at test time.
  set(MIPS_SYSROOT       "/usr/mips64-linux-gnuabi64")
endif()

# -EB makes the big-endian intent explicit (it is the default for this
# compiler, but we state it so the flag set is unambiguous).
set(MIPS_COMPILER_FLAGS "\
    -march=mips64r2 -mabi=64 -EB")

set(MIPS_QEMU_CPU_FLAGS "MIPS64R2-generic")

set(MIPS64_TEST_DEFAULT_LLVM_FLAGS
  ""
  CACHE INTERNAL "Default llvm codegen flags for testing purposes")

set(CMAKE_C_FLAGS_INIT   "${MIPS_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${MIPS_COMPILER_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${MIPS_COMPILER_FLAGS}")
