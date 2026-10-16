# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Toolchain file for bare-metal riscv-none-elf GCC/newlib cross-compilation,
# using one sysroot for both compilation and linking.
#
# samples/baremetal_riscv64 requires newlib to be built with -mcmodel=medany
# because QEMU's virt machine places RAM at 0x80000000. The toolchain must also
# provide semihost.specs for console I/O and process exit.

cmake_minimum_required(VERSION 3.26)

# CMake invokes the toolchain file twice during the first build, but only once
# during subsequent rebuilds. This was causing the various flags to be added
# twice on the first build, and on a rebuild ninja would see only one set of
# the flags and rebuild the world.
# https://github.com/android-ndk/ndk/issues/323
if(RISCV_TOOLCHAIN_INCLUDED)
  return()
endif()
set(RISCV_TOOLCHAIN_INCLUDED true)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT "${RISCV_TOOLCHAIN_ROOT}" STREQUAL "")
  set(CMAKE_AR           "${RISCV_TOOLCHAIN_ROOT}/bin/riscv-none-elf-ar")
  set(CMAKE_C_COMPILER   "${RISCV_TOOLCHAIN_ROOT}/bin/riscv-none-elf-gcc")
  set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv-none-elf-g++")
  set(CMAKE_RANLIB       "${RISCV_TOOLCHAIN_ROOT}/bin/riscv-none-elf-ranlib")
  set(CMAKE_STRIP        "${RISCV_TOOLCHAIN_ROOT}/bin/riscv-none-elf-strip")
endif()

# Bare-metal: configure-time link checks have no crt0/syscalls.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_EXTENSIONS OFF CACHE BOOL "" FORCE) # gnu17 selects a clock_nanosleep() path newlib lacks.
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_POSIX OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_THREADING OFF CACHE BOOL "" FORCE)
set(IREE_SYNCHRONIZATION_DISABLE_UNSAFE ON CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_TASK OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY OFF CACHE BOOL "" FORCE)

# The default RISC-V ISA specification treats zicsr and zifencei as
# separate extensions, so name them explicitly.
set(RISCV_COMPILER_FLAGS "\
    -march=rv64imafdc_zicsr_zifencei -mabi=lp64d -mcmodel=medany \
    -DIREE_PLATFORM_GENERIC=1 \
    -DIREE_FILE_IO_ENABLE=0 -DIREE_TIME_NOW_FN=\"\{ return 0; \}\" -DIREE_DEVICE_SIZE_T=uint64_t -DPRIdsz=PRIu64")

set(CMAKE_C_FLAGS_INIT   "${RISCV_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${RISCV_COMPILER_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${RISCV_COMPILER_FLAGS}")

# GNUInstallDirs does not set CMAKE_INSTALL_LIBDIR in this configuration.
set(CMAKE_INSTALL_LIBDIR "lib" CACHE PATH "")
