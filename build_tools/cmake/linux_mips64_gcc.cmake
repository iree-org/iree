# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26) # minimum CMake version

# CMake invokes the toolchain file twice during the first build, but only once
# during subsequent rebuilds. This was causing the various flags to be added
# twice on the first build, and on a rebuild ninja would see only one set of the
# flags and rebuild the world.
# https://github.com/android-ndk/ndk/issues/323

if(MIPS_TOOLCHAIN_INCLUDED)
  return()
endif()
set(MIPS_TOOLCHAIN_INCLUDED true)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR mips64)

set(MIPS_TOOLCHAIN_PREFIX "${MIPS_TOOLCHAIN_PREFIX}")
if(MIPS_TOOLCHAIN_PREFIX STREQUAL "")
  set(MIPS_TOOLCHAIN_PREFIX "mips64el-linux-gnuabi64-")
endif()

if(NOT "${MIPS_TOOLCHAIN_ROOT}" STREQUAL "")   # use the GNU toolchain binaries under the provided root
  set(CMAKE_AR           "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}ar")
  set(CMAKE_C_COMPILER   "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}gcc")
  set(CMAKE_CXX_COMPILER "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}g++")
  set(CMAKE_RANLIB       "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}ranlib")
  set(CMAKE_STRIP        "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}strip")
  set(CMAKE_SYSROOT      "${MIPS_TOOLCHAIN_ROOT}/sysroot")
  # Used only by run_mips_test.sh (QEMU -L) at test time, kept separate from
  # CMAKE_SYSROOT because passing --sysroot to gcc breaks the cross-compile
  # when using the distro-packaged toolchain
  set(MIPS_SYSROOT       "${MIPS_TOOLCHAIN_ROOT}/sysroot")
else()
  set(CMAKE_C_COMPILER   "${MIPS_TOOLCHAIN_PREFIX}gcc")
  set(CMAKE_CXX_COMPILER "${MIPS_TOOLCHAIN_PREFIX}g++")
  # Do NOT set CMAKE_SYSROOT here: the distro mips64el-linux-gnuabi64-gcc
  # package already knows its own sysroot, and passing --sysroot explicitly
  # breaks the build. MIPS_SYSROOT below is only for QEMU's -L flag at test
  # time (see run_mips_test.sh / iree_cc_test.cmake).
  set(MIPS_SYSROOT       "/usr/mips64el-linux-gnuabi64")
endif()

set(MIPS_COMPILER_FLAGS "\
    -march=mips64r2 -mabi=64")

set(MIPS_QEMU_CPU_FLAGS "MIPS64R2-generic") # Generic 64-bit MIPS Release 2 CPU model == mips64r2


# The block below is the native (llvm-cpu) codegen version and is commented out on purpose:
# this VMVX-only build does no MIPS code generation, so there are no llvm-cpu
# codegen flags to pass. Uncomment and populate it when native MIPS
# codegen is added; until then the active value below must stay empty.
# set(MIPS64_TEST_DEFAULT_LLVM_FLAGS
#   "--iree-llvmcpu-target-triple=mips64el-unknown-linux-gnu"   #<arch>-<vendor>-<os>-<environment>
#   "--iree-llvmcpu-target-abi=64"
#   "--iree-llvmcpu-target-cpu=mips64"
#   CACHE INTERNAL "Default llvm codegen flags for testing purposes")
set(MIPS64_TEST_DEFAULT_LLVM_FLAGS
  ""
  CACHE INTERNAL "Default llvm codegen flags for testing purposes")


set(CMAKE_C_FLAGS_INIT   "${MIPS_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${MIPS_COMPILER_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${MIPS_COMPILER_FLAGS}")
