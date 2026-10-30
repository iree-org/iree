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
# are Release 6, which also mandates NaN-2008 floating point. The distro
# mips64el-linux-gnuabi64 toolchain ships only an r2 / legacy-NaN n64 sysroot
# and therefore cannot build or run MSA code. This file instead uses the
# Codescape IMG GNU toolchain (prefix mips-img-linux-gnu-, GCC 11.2+), which
# ships the mips64r6 / n64 / little-endian / NaN-2008 multilib + sysroot
# (`mipsel-r6-hard`). Validated end-to-end under qemu-mips64el -cpu I6400.
#
# Configure with, e.g.:
#   cmake -G Ninja -B build-mips-r6 \
#     -DCMAKE_TOOLCHAIN_FILE=build_tools/cmake/linux_mips64r6_gcc.cmake \
#     -DMIPS_TOOLCHAIN_ROOT=$HOME/toolchains/mips-img-linux-gnu/2024.10-03 \
#     -DIREE_BUILD_COMPILER=OFF -DIREE_HOST_BIN_DIR=$(realpath build/tools) ...
#
# Overridable: MIPS_TOOLCHAIN_PREFIX, MIPS_R6_SYSROOT_VARIANT.

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

# The Codescape toolchain is not on PATH and has no default location, so its
# root must be provided explicitly.
if("${MIPS_TOOLCHAIN_ROOT}" STREQUAL "")
  message(FATAL_ERROR
    "linux_mips64r6_gcc.cmake requires -DMIPS_TOOLCHAIN_ROOT=<toolchain root>, "
    "i.e. the directory containing bin/mips-img-linux-gnu-gcc and sysroot/ "
    "(the extracted Codescape IMG GNU/Linux package, e.g. "
    ".../mips-img-linux-gnu/2024.10-03).")
endif()

set(MIPS_TOOLCHAIN_PREFIX "${MIPS_TOOLCHAIN_PREFIX}")
if(MIPS_TOOLCHAIN_PREFIX STREQUAL "")
  set(MIPS_TOOLCHAIN_PREFIX "mips-img-linux-gnu-")
endif()

# Which per-multilib sysroot to hand QEMU's loader (-L) at test time. The
# Codescape sysroot is split per multilib; mipsel-r6-hard matches the
# little-endian r6 hard-float NaN-2008 flags set below.
set(MIPS_R6_SYSROOT_VARIANT "${MIPS_R6_SYSROOT_VARIANT}")
if(MIPS_R6_SYSROOT_VARIANT STREQUAL "")
  set(MIPS_R6_SYSROOT_VARIANT "mipsel-r6-hard")
endif()

# CMake runs its compiler checks in throwaway try_compile sub-projects that
# re-include this toolchain file but do NOT inherit the -D variables from the
# main configure. Forward the ones we need so the sub-project can locate the
# toolchain (otherwise the MIPS_TOOLCHAIN_ROOT check above fails there and
# CMAKE_C_COMPILER never gets set).
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  MIPS_TOOLCHAIN_ROOT
  MIPS_TOOLCHAIN_PREFIX
  MIPS_R6_SYSROOT_VARIANT)

set(CMAKE_AR           "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}ar")
set(CMAKE_C_COMPILER   "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}g++")
set(CMAKE_RANLIB       "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}ranlib")
set(CMAKE_STRIP        "${MIPS_TOOLCHAIN_ROOT}/bin/${MIPS_TOOLCHAIN_PREFIX}strip")

# Do NOT set CMAKE_SYSROOT. The Codescape gcc selects the correct per-multilib
# sysroot automatically from the -march/-mabi/-EL flags below (verified:
# `gcc <flags> -print-sysroot` returns .../sysroot/mipsel-r6-hard). Passing
# --sysroot pointed at the parent sysroot/ dir would break library resolution,
# because that dir is a container of per-multilib subdirs, not a single
# sysroot. MIPS_SYSROOT is used only for QEMU's -L flag at test time (see
# run_mips_test.sh / iree_cc_test.cmake).
set(MIPS_SYSROOT "${MIPS_TOOLCHAIN_ROOT}/sysroot/${MIPS_R6_SYSROOT_VARIANT}")

# MIPS64 Release 6 + n64 ABI + little-endian. r6 implies NaN-2008, so the
# binaries load on NaN-2008-only MSA cores (QEMU -cpu I6400). MSA itself is
# only needed by the compute kernels (emitted by llvm-cpu codegen below), not
# by the runtime glue, so it is deliberately left off the runtime C/C++ flags.
set(MIPS_COMPILER_FLAGS "\
    -march=mips64r6 -mabi=64 -EL")

# The Codescape GCC 11.2 emits false-positive buffer overread/overflow
# diagnostics on third-party (vendored) flatcc's flatbuffers pointer arithmetic
# at -O3 -- e.g. pmemaccess.h `mem_read_32` "reading 4 bytes from a region of
# size 0". These are not real bugs and must not be patched in flatcc, so they
# are silenced for the whole r6 runtime build. (The r2 distro GCC 12 did not
# emit them, which is why IREE's shared warning-suppression list does not cover
# them.) Kept in the same family to avoid rediscovering each across rebuilds.
set(MIPS_COMPILER_FLAGS "${MIPS_COMPILER_FLAGS} \
    -Wno-stringop-overread -Wno-stringop-overflow -Wno-array-bounds")

set(MIPS_QEMU_CPU_FLAGS "I6400") # MIPS64r6 core WITH MSA (MIPS64R2-generic lacks MSA)

# Marks this as an MSA-capable (r6) MIPS build. Consumed by
# iree_is_bytecode_module_test_excluded_by_labels (iree_check_test.cmake): on a
# scalar r2 build the data-tiling/winograd suites are `nomips`-excluded because
# they need SIMD, but on r6/MSA they compile and run, so the exclusion is lifted
# when this is set. Leaves the scalar r2 build (linux_mips64_gcc.cmake) unchanged.
set(IREE_MIPS_ENABLE_MSA ON CACHE INTERNAL "MIPS MSA (r6) build")

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
