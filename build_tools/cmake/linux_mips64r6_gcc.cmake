# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# MIPS64 Release 6 + MSA (128-bit SIMD) cross-compilation toolchain.
#
# This is the r6/MSA counterpart to linux_mips64_gcc.cmake (which targets the
# generic mips64r2 / legacy-NaN distro toolchain and is the scalar baseline
# gate). It is kept as a SEPARATE file so the r2 build stays untouched.
#
# MSA is a MIPS Release 5+ extension; real MSA-capable cores (e.g. the I6400)
# are Release 6, which also mandates NaN-2008 floating point. The generic
# mips64el-linux-gnuabi64 distro toolchain ships only an r2 / legacy-NaN n64
# sysroot and therefore cannot build or run MSA code. This file instead uses
# the Debian/Ubuntu-packaged r6 toolchain, prefix mipsisa64r6el-linux-gnuabi64-
# (apt packages gcc-mipsisa64r6el-linux-gnuabi64, g++-..., and the sysroot
# libc6-dev-mips64r6el-cross), which ships the mips64r6 / n64 / little-endian /
# NaN-2008 compiler + sysroot. Validated end-to-end under qemu-mips64el
# -cpu I6400. MSA itself is emitted into the llvm-cpu .so by LLVM (not by this
# GCC), so the GCC only has to produce r6/NaN-2008 runtime glue that loads on an
# I6400 core; it does not need an MSA-enabled multilib.
#
# Install the toolchain with:
#   sudo apt install gcc-mipsisa64r6el-linux-gnuabi64 \
#     g++-mipsisa64r6el-linux-gnuabi64 libc6-dev-mips64r6el-cross qemu-user
#
# Configure with, e.g.:
#   cmake -G Ninja -B build-mips-r6 \
#     -DCMAKE_TOOLCHAIN_FILE=build_tools/cmake/linux_mips64r6_gcc.cmake \
#     -DIREE_BUILD_COMPILER=OFF -DIREE_HOST_BIN_DIR=$(realpath build/tools) ...
#
# Overridable: MIPS_TOOLCHAIN_PREFIX, MIPS_TOOLCHAIN_ROOT, MIPS_SYSROOT.

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

# The distro cross toolchain installs as mipsisa64r6el-linux-gnuabi64-* on PATH
# (in /usr/bin), with the sysroot under /usr/mipsisa64r6el-linux-gnuabi64. The
# prefix is overridable; MIPS_TOOLCHAIN_ROOT may point at a non-PATH toolchain.
set(MIPS_TOOLCHAIN_PREFIX "${MIPS_TOOLCHAIN_PREFIX}")
if(MIPS_TOOLCHAIN_PREFIX STREQUAL "")
  set(MIPS_TOOLCHAIN_PREFIX "mipsisa64r6el-linux-gnuabi64-")
endif()

# CMake runs its compiler checks in throwaway try_compile sub-projects that
# re-include this toolchain file but do NOT inherit the -D variables from the
# main configure. Forward the ones we need.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  MIPS_TOOLCHAIN_PREFIX
  MIPS_TOOLCHAIN_ROOT
  MIPS_SYSROOT)

if(NOT "${MIPS_TOOLCHAIN_ROOT}" STREQUAL "")
  # Use the GNU toolchain binaries under the provided root.
  set(CMAKE_AR           "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}ar")
  set(CMAKE_C_COMPILER   "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}gcc")
  set(CMAKE_CXX_COMPILER "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}g++")
  set(CMAKE_RANLIB       "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}ranlib")
  set(CMAKE_STRIP        "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}strip")
else()
  # Toolchain is on PATH (the apt install case).
  set(CMAKE_AR           "${MIPS_TOOLCHAIN_PREFIX}ar")
  set(CMAKE_C_COMPILER   "${MIPS_TOOLCHAIN_PREFIX}gcc")
  set(CMAKE_CXX_COMPILER "${MIPS_TOOLCHAIN_PREFIX}g++")
  set(CMAKE_RANLIB       "${MIPS_TOOLCHAIN_PREFIX}ranlib")
  set(CMAKE_STRIP        "${MIPS_TOOLCHAIN_PREFIX}strip")
endif()

# The distro gcc is configured for this triple and finds its own r6 sysroot
# (/usr/mipsisa64r6el-linux-gnuabi64, from libc6-dev-mips64r6el-cross)
# automatically, so CMAKE_SYSROOT is deliberately left unset. MIPS_SYSROOT is
# used only for QEMU's -L flag / QEMU_LD_PREFIX at test time (see
# run_mips_test.sh / iree_cc_test.cmake).
set(MIPS_SYSROOT "${MIPS_SYSROOT}")
if(MIPS_SYSROOT STREQUAL "")
  # Strip the trailing dash from the prefix to get the sysroot dir name.
  string(REGEX REPLACE "-$" "" _MIPS_TRIPLE "${MIPS_TOOLCHAIN_PREFIX}")
  set(MIPS_SYSROOT "/usr/${_MIPS_TRIPLE}")
endif()

# mipsisa64r6el already implies mips64r6 / n64 / little-endian / NaN-2008, so
# these are the compiler's defaults; set them explicitly anyway so the intent
# is unambiguous and the build is correct even under a differently-configured
# gcc of the same name.
set(MIPS_COMPILER_FLAGS "\
    -march=mips64r6 -mabi=64 -EL")

# NOTE: the distro GCC 12 is not expected to emit the false-positive flatcc
# stringop-overread/overflow/array-bounds diagnostics at -O3 (matching the r2
# distro GCC 12). If a future GCC version regresses and -Werror breaks the
# flatcc build, add the -Wno-stringop-overread/-overflow/-array-bounds flags
# here.

set(MIPS_QEMU_CPU_FLAGS "I6400") # MIPS64r6 core WITH MSA (MIPS64R2-generic lacks MSA)

# Marks this as an MSA-capable (r6) MIPS build. Consumed by
# iree_is_bytecode_module_test_excluded_by_labels (iree_check_test.cmake): on a
# scalar r2 build the data-tiling/winograd suites are `nomips`-excluded because
# they need SIMD, but on r6/MSA they compile and run, so the exclusion is lifted
# when this is set. Leaves the scalar r2 build (linux_mips64_gcc.cmake) unchanged.
set(IREE_MIPS_ENABLE_MSA ON CACHE INTERNAL "MIPS MSA (r6) build")

# This cross-build compiles test modules with a prebuilt host compiler
# (IREE_HOST_BIN_DIR), which is typically built lean (llvm-cpu/vmvx only). The
# runtime-side IREE_TARGET_BACKEND_* options do not describe that compiler, so
# without this, test modules for GPU backends (vulkan/cuda/...) the host
# compiler lacks would be generated and fail to compile. Restrict test-module
# generation to the backends the lean host compiler supports. Override with
# -DIREE_TEST_TARGET_BACKENDS=... (or clear it) if your host compiler has more.
set(IREE_TEST_TARGET_BACKENDS "llvm-cpu;vmvx" CACHE STRING
    "Host-compiler target-backend allowlist for test-module generation")

# The samples under samples/simple_embedding/ embed a precompiled module and
# load it with the embedded-ELF loader, dispatching on a hardcoded per-arch
# switch (x86_64/riscv/arm_64) that #errors on any other target. MIPS has no
# arm there, and the embedded-ELF loader itself is still a stub on MIPS
# (hal/local/elf/arch/mips_64.c), so these samples cannot build or run here
# yet. They are demos, not part of the test gate, so disable samples for this
# cross-build. Override with -DIREE_BUILD_SAMPLES=ON once the embedded-ELF MIPS
# path lands.
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "Disabled for the MIPS cross-build")

# Native (llvm-cpu) codegen flags injected into llvm-cpu test compiles for
# mips_64 (see iree_compile_flags_for_platform in iree_macros.cmake). MIPS uses
# the system dynamic library loader (link_embedded=false); the embedded ELF
# loader would need real MIPS relocations in hal/local/elf/arch/mips_64.c
# (still a stub).
#
# BOTH cpu and feature flags are required, and they do different jobs:
#  * --iree-llvmcpu-target-cpu=i6400 makes LLVM emit a Release-6 / NaN-2008 ELF
#    (`readelf -h` flags: nan2008, mips64r6). Without it the .so is flagged r2 /
#    legacy-NaN and the I6400 core refuses to load it.
#  * --iree-llvmcpu-target-cpu-features=+msa actually enables 128-bit
#    vectorization (native_vector_size 4 -> 16). i6400 implies MSA in LLVM's
#    subtarget, but that implied feature does not appear in the feature string
#    IREE inspects to set the vector width, so +msa must be passed explicitly.
set(MIPS64_TEST_DEFAULT_LLVM_FLAGS
  "--iree-llvmcpu-target-triple=mips64el-linux-gnuabi64"
  "--iree-llvmcpu-link-embedded=false"
  "--iree-llvmcpu-target-cpu=i6400"
  "--iree-llvmcpu-target-cpu-features=+msa"
  CACHE INTERNAL "Default llvm codegen flags for testing purposes")

set(CMAKE_C_FLAGS_INIT   "${MIPS_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${MIPS_COMPILER_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${MIPS_COMPILER_FLAGS}")
