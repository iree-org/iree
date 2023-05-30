# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if (NOT (IREE_ARCH STREQUAL "arm_64"))
  return()
endif()

iree_select_compiler_opts(IREE_UK_COPTS_ARM_64_DOTPROD
  CLANG_OR_GCC
    "-march=armv8.2-a+dotprod"
)

iree_select_compiler_opts(IREE_UK_COPTS_ARM_64_I8MM
  CLANG_OR_GCC
    "-march=armv8.2-a+i8mm"
)

iree_cc_library(
  NAME
    common_arm_64
  HDRS
    "common_arm_64.h"
  DEPS
    iree::builtins::ukernel::internal_headers
    iree::schemas::cpu_data
)

iree_cc_library(
  NAME
    arm_64_dotprod
  SRCS
    "mmt4d_arm_64_dotprod.c"
  COPTS
    "${IREE_UK_COPTS_ARM_64_DOTPROD}"
  DEPS
    iree::builtins::ukernel::internal_headers
)

iree_cc_library(
  NAME
    arm_64_i8mm
  SRCS
    "mmt4d_arm_64_i8mm.c"
  COPTS
    "${IREE_UK_COPTS_ARM_64_I8MM}"
  DEPS
    iree::builtins::ukernel::internal_headers
)

iree_cc_library(
  NAME
    arm_64
  SRCS
    "mmt4d_arm_64.c"
    "pack_arm_64.c"
    "query_tile_sizes_arm_64.c"
    "unpack_arm_64.c"
  DEPS
    ::common_arm_64
    ::arm_64_dotprod
    ::arm_64_i8mm
    iree::base::core_headers
    iree::schemas::cpu_data
    iree::builtins::ukernel::internal_headers
    ${IREE_UK_ARM_64_DEPS}
  PUBLIC
)
