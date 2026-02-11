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

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT "${RISCV_TOOLCHAIN_ROOT}" STREQUAL "")
  set(CMAKE_AR           "${RISCV_TOOLCHAIN_ROOT}/bin/${RISCV_TOOLCHAIN_PREFIX}llvm-ar")
  set(CMAKE_C_COMPILER   "${RISCV_TOOLCHAIN_ROOT}/bin/${RISCV_TOOLCHAIN_PREFIX}clang")
  set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/${RISCV_TOOLCHAIN_PREFIX}clang++")
  set(CMAKE_RANLIB       "${RISCV_TOOLCHAIN_ROOT}/bin/${RISCV_TOOLCHAIN_PREFIX}llvm-ranlib")
  set(CMAKE_STRIP        "${RISCV_TOOLCHAIN_ROOT}/bin/${RISCV_TOOLCHAIN_PREFIX}llvm-strip")
  set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot")
endif()

# Specify ISA spec for march=rv64gc. This is to resolve the mismatch between
# llvm and binutil ISA version.
set(RISCV_COMPILER_FLAGS "\
    -march=rv64i2p1ma2p1f2p2d2p2c2p0 -mabi=lp64d")
set(RISCV64_TEST_DEFAULT_LLVM_FLAGS
  "--iree-llvmcpu-target-triple=riscv64"
  "--iree-llvmcpu-target-abi=lp64d"
  "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+zvl512b,+v"
  "--riscv-v-fixed-length-vector-lmul-max=8"
  CACHE INTERNAL "Default llvm codegen flags for testing purposes")

set(CMAKE_C_FLAGS_INIT   "${RISCV_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${RISCV_COMPILER_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${RISCV_COMPILER_FLAGS}")
