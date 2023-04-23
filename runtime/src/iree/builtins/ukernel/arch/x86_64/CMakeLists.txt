# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Target CPUs supporting AVX2+FMA3. That includes Intel Haswell (2013) and newer
# and AMD Excavator (2015) and newer. There is no current plan to look after
# SIMD performance on x86 microarchitectures not supporting this.
iree_select_compiler_opts(IREE_UK_COPTS_X86_64_AVX2_FMA
  CLANG_OR_GCC
    "-mavx2"
    "-mfma"
  MSVC
    "/arch:AVX2"
)

# TODO: Support AVX-VNNI and/or AVX-VNNI-INT8?
# 1. AVX-VNNI is the 256-bit backport of AVX-512-VNNI. It is supported in Alder
#    Lake (2022).
# 2  AVX-VNNI-INT8 is a future ISA extension not yet mentioned in the Intel SDM,
#    but we can glean the following information from these links: it will be
#    supported in Sierra Forest (2024) it will introduce VPDPBSSD, the missing
#    signed*signed counterpart to VNNI's VPDPBUSD, meaning we will finally be
#    able to use the 8bit path, so that VNNI-IN8 will be a >= 2x win for us over
#    VNNI.
#    https://en.wikipedia.org/wiki/Sierra_Forest
#    https://gcc.gnu.org/pipermail/gcc-patches/2022-October/603546.html
#    https://gcc.gnu.org/git/?p=gcc.git;a=commitdiff;h=406675947d26ccbc2108e9689a2918bb36f61a63


# Target CPUs supporting the typical baseline AVX-512 features (F+VL+CD+BW+DQ).
# That includes Intel Skylake/server (2017) and AMD Zen4 (2022) while recent
# non-server x86 microarchitectures not supporting this continue to exist.
iree_select_compiler_opts(IREE_UK_COPTS_X86_64_AVX512_BASE
  CLANG_OR_GCC
    "-mavx512f"
    "-mavx512vl"
    "-mavx512cd"
    "-mavx512bw"
    "-mavx512dq"
  MSVC
    "/arch:AVX512"
)

# Target CPUs supporting the AVX-512 VNNI feature. That includes Intel Sunny
# Cove (2019) and newer, and AMD Zen4 (2022).
iree_select_compiler_opts(IREE_UK_COPTS_X86_64_AVX512_VNNI_RELATIVE
  CLANG_OR_GCC
    "-mavx512vnni"
  MSVC
)
set(IREE_UK_COPTS_X86_64_AVX512_VNNI
  "${IREE_UK_COPTS_X86_64_AVX512_BASE}"
  "${IREE_UK_COPTS_X86_64_AVX512_VNNI_RELATIVE}"
)

check_cxx_compiler_flag("${IREE_UK_COPTS_X86_64_AVX2_FMA}" IREE_UK_BUILD_X86_64_AVX2_FMA)
check_cxx_compiler_flag("${IREE_UK_COPTS_X86_64_AVX512_BASE}" IREE_UK_BUILD_X86_64_AVX512_BASE)
check_cxx_compiler_flag("${IREE_UK_COPTS_X86_64_AVX512_VNNI}" IREE_UK_BUILD_X86_64_AVX512_VNNI)

configure_file(config.h.in config.h)

iree_cc_library(
  NAME
    config
  HDRS
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

iree_cc_library(
  NAME
    common_x86_64
  HDRS
    "common_x86_64.h"
  DEPS
    iree::builtins::ukernel::internal_headers
    iree::schemas::cpu_data
)

if(IREE_UK_BUILD_X86_64_AVX2_FMA)
  iree_cc_library(
    NAME
      x86_64_avx2_fma
    SRCS
      "mmt4d_x86_64_avx2_fma.c"
      "pack_x86_64_avx2_fma.c"
      "unpack_x86_64_avx2_fma.c"
    COPTS
      "${IREE_UK_COPTS_X86_64_AVX2_FMA}"
    DEPS
      iree::builtins::ukernel::internal_headers
  )
  list(APPEND IREE_UK_X86_64_DEPS "iree::builtins::ukernel::arch::x86_64::x86_64_avx2_fma")
endif()

if(IREE_UK_BUILD_X86_64_AVX512_BASE)
  iree_cc_library(
    NAME
      x86_64_avx512_base
    SRCS
      "mmt4d_x86_64_avx512_base.c"
      "pack_x86_64_avx512_base.c"
      "unpack_x86_64_avx512_base.c"
    COPTS
      "${IREE_UK_COPTS_X86_64_AVX512_BASE}"
    DEPS
      iree::builtins::ukernel::internal_headers
  )
  list(APPEND IREE_UK_X86_64_DEPS "iree::builtins::ukernel::arch::x86_64::x86_64_avx512_base")
endif()

if(IREE_UK_BUILD_X86_64_AVX512_VNNI)
  iree_cc_library(
    NAME
      x86_64_avx512_vnni
    SRCS
      "mmt4d_x86_64_avx512_vnni.c"
    COPTS
      "${IREE_UK_COPTS_X86_64_AVX512_VNNI}"
    DEPS
      iree::builtins::ukernel::internal_headers
  )
  list(APPEND IREE_UK_X86_64_DEPS "iree::builtins::ukernel::arch::x86_64::x86_64_avx512_vnni")
endif()


iree_cc_library(
  NAME
    x86_64
  HDRS
    "mmt4d_x86_64.h"
    "pack_x86_64.h"
    "query_tile_sizes_x86_64.h"
    "unpack_x86_64.h"
  SRCS
    "mmt4d_x86_64.c"
    "pack_x86_64.c"
    "query_tile_sizes_x86_64.c"
    "unpack_x86_64.c"
  DEPS
    ::common_x86_64
    iree::base::core_headers
    iree::builtins::ukernel::internal_headers
    ${IREE_UK_X86_64_DEPS}
  PUBLIC
)
