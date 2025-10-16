# Copyright 2020 The IREE Authors
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
endif(RISCV_TOOLCHAIN_INCLUDED)
set(RISCV_TOOLCHAIN_INCLUDED true)

set(CMAKE_SYSTEM_PROCESSOR riscv)

set(RISCV_HOST_TAG linux)

set(RISCV_TOOL_PATH "$ENV{HOME}/riscv" CACHE PATH "RISC-V tool path")

set(RISCV_TOOLCHAIN_ROOT "${RISCV_TOOL_PATH}/toolchain/clang/${RISCV_HOST_TAG}/RISCV" CACHE PATH "RISC-V compiler path")
set(RISCV_TOOLCHAIN_PREFIX "riscv64-unknown-linux-gnu-" CACHE STRING "RISC-V toolchain prefix")
set(CMAKE_FIND_ROOT_PATH ${RISCV_TOOLCHAIN_ROOT})
list(APPEND CMAKE_PREFIX_PATH "${RISCV_TOOLCHAIN_ROOT}")

set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang++")
set(CMAKE_AR "${RISCV_TOOLCHAIN_ROOT}/bin/llvm-ar")
set(CMAKE_RANLIB "${RISCV_TOOLCHAIN_ROOT}/bin/llvm-ranlib")
set(CMAKE_STRIP "${RISCV_TOOLCHAIN_ROOT}/bin/llvm-strip")

set(RISCV_COMPILER_FLAGS)
set(RISCV_COMPILER_FLAGS_CXX)
set(RISCV_LINKER_FLAGS)
set(RISCV_LINKER_FLAGS_EXE "" CACHE STRING "Linker flags for RISCV executables")

set(CMAKE_SYSTEM_NAME Linux)

if(RISCV_CPU MATCHES "riscv_64")
  set(CMAKE_SYSTEM_PROCESSOR riscv64)
elseif(RISCV_CPU MATCHES "riscv_32")
  set(CMAKE_SYSTEM_PROCESSOR riscv32)
endif()

if(RISCV_CPU STREQUAL "linux-riscv_64")
  set(CMAKE_SYSTEM_LIBRARY_PATH "${RISCV_TOOLCHAIN_ROOT}/sysroot/lib64/lp64d")
  set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot")
  # Specify ISP spec for march=rv64gc. This is to resolve the mismatch between
  # llvm and binutil ISA version.
  set(RISCV_COMPILER_FLAGS "${RISCV_COMPILER_FLAGS} \
      -march=rv64i2p1ma2p1f2p2d2p2c2p0 -mabi=lp64d")
  set(RISCV_LINKER_FLAGS "${RISCV_LINKER_FLAGS} -lstdc++ -lpthread -lm -ldl")
  set(RISCV64_TEST_DEFAULT_LLVM_FLAGS
    "--iree-llvmcpu-target-triple=riscv64"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+zvl512b,+v"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    CACHE INTERNAL "Default llvm codegen flags for testing purposes")
elseif(RISCV_CPU STREQUAL "linux-riscv_32")
  list(APPEND CMAKE_SYSTEM_LIBRARY_PATH
    "${RISCV_TOOLCHAIN_ROOT}/sysroot/usr/lib32"
    "${RISCV_TOOLCHAIN_ROOT}/sysroot/usr/lib32/ilp32d"
  )
  set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot")
  # Specify ISP spec for march=rv32gc. This is to resolve the mismatch between
  # llvm and binutil ISA version.
  set(RISCV_COMPILER_FLAGS "${RISCV_COMPILER_FLAGS} \
      -march=rv32i2p1ma2p1f2p2d2p2c2p0 -mabi=ilp32d \
      -Wno-atomic-alignment")
  set(RISCV_LINKER_FLAGS "${RISCV_LINKER_FLAGS} -lstdc++ -lpthread -lm -ldl -latomic")
  set(RISCV32_TEST_DEFAULT_LLVM_FLAGS
    "--iree-llvmcpu-target-triple=riscv32"
    "--iree-llvmcpu-target-abi=ilp32d"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl512b,+zve32f"
    "--riscv-v-fixed-length-vector-lmul-max=8"
    CACHE INTERNAL "Default llvm codegen flags for testing purposes")
endif()

set(CMAKE_C_FLAGS             "${RISCV_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS           "${RISCV_COMPILER_FLAGS} ${RISCV_COMPILER_FLAGS_CXX} ${CMAKE_CXX_FLAGS}")
set(CMAKE_ASM_FLAGS           "${RISCV_COMPILER_FLAGS} ${CMAKE_ASM_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${RISCV_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${RISCV_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS    "${RISCV_LINKER_FLAGS} ${RISCV_LINKER_FLAGS_EXE} ${CMAKE_EXE_LINKER_FLAGS}")
