# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

###############################################################################
# configuration
###############################################################################

check_cxx_compiler_flag("-march=armv8.2-a+dotprod" IREE_UK_BUILD_ARM_64_DOTPROD)
check_cxx_compiler_flag("-march=armv8.2-a+i8mm" IREE_UK_BUILD_ARM_64_I8MM)
configure_file(config.h.in config.h)

iree_cc_library(
  NAME
    assembly
  HDRS
    "assembly.h"
)

###############################################################################
# mmt4d tile funcs
###############################################################################

iree_cc_library(
  NAME
    mmt4d_tile_arm_64
  HDRS
    "mmt4d_tile_arm_64.h"
  SRCS
    "mmt4d_tile_arm_64.S"
  DEPS
    ::assembly
    iree::builtins::ukernel::exported_bits
)
list(APPEND IREE_UK_MMT4D_TILE_ARM_64_DEPS "iree::builtins::ukernel::arch::arm_64::mmt4d_tile_arm_64")

if(IREE_UK_BUILD_ARM_64_DOTPROD)
  iree_cc_library(
    NAME
      mmt4d_tile_arm_64_dotprod
    HDRS
      "mmt4d_tile_arm_64.h"
    SRCS
      "mmt4d_tile_arm_64_dotprod.S"
    COPTS
      "-march=armv8.2-a+dotprod"
    DEPS
      ::assembly
      iree::builtins::ukernel::exported_bits

  )
  list(APPEND IREE_UK_MMT4D_TILE_ARM_64_DEPS "iree::builtins::ukernel::arch::arm_64::mmt4d_tile_arm_64_dotprod")
endif()

if(IREE_UK_BUILD_ARM_64_I8MM)
  iree_cc_library(
    NAME
      mmt4d_tile_arm_64_i8mm
    HDRS
      "mmt4d_tile_arm_64.h"
    SRCS
      "mmt4d_tile_arm_64_i8mm.S"
    COPTS
      "-march=armv8.2-a+i8mm"
    DEPS
      ::assembly
      iree::builtins::ukernel::exported_bits
  )
  list(APPEND IREE_UK_MMT4D_TILE_ARM_64_DEPS "iree::builtins::ukernel::arch::arm_64::mmt4d_tile_arm_64_i8mm")
endif()

###############################################################################
# mmt4d entry point
###############################################################################

iree_cc_library(
  NAME
    mmt4d_arm_64
  HDRS
    "mmt4d_arm_64.h"
  SRCS
    "mmt4d_arm_64.c"
  DEPS
    iree::base::core_headers
    iree::schemas::cpu_data
    iree::builtins::ukernel::headers
    ${IREE_UK_MMT4D_TILE_ARM_64_DEPS}
  PUBLIC
)

###############################################################################
# pack tile funcs
###############################################################################

iree_cc_library(
  NAME
    pack_tile_arm_64
  HDRS
    "pack_tile_arm_64.h"
  SRCS
    "pack_tile_arm_64.c"
  DEPS
    iree::builtins::ukernel::exported_bits
)

###############################################################################
# pack entry point
###############################################################################

iree_cc_library(
  NAME
    pack_arm_64
  HDRS
    "pack_arm_64.h"
  SRCS
    "pack_arm_64.c"
  DEPS
    iree::base::core_headers
    iree::schemas::cpu_data
    iree::builtins::ukernel::headers
    ::pack_tile_arm_64
  PUBLIC
)

###############################################################################
# query_tile_sizes entry point
###############################################################################

iree_cc_library(
  NAME
  query_tile_sizes_arm_64
  HDRS
    "query_tile_sizes_arm_64.h"
  SRCS
    "query_tile_sizes_arm_64.c"
  DEPS
    iree::base::core_headers
    iree::schemas::cpu_data
    iree::builtins::ukernel::headers
  PUBLIC
)
