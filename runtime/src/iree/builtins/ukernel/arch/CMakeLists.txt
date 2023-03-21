# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

option(IREE_UK_FORCE_DISABLE_ARCH_SPECIFIC_CODE "Disable all architecture-specific code in builtin kernels" OFF)

if(IREE_UK_FORCE_DISABLE_ARCH_SPECIFIC_CODE)
  set(IREE_UK_ENABLE_ARCH_SPECIFIC_CODE FALSE)
else()
  set(IREE_UK_ENABLE_ARCH_SPECIFIC_CODE TRUE)
endif()

if(IREE_UK_ENABLE_ARCH_SPECIFIC_CODE)
  if (IREE_ARCH STREQUAL "arm_64")
    set(IREE_UK_ARCH_ARM_64 TRUE)
    add_subdirectory(arm_64)
    list(APPEND IREE_UK_ARCH_DEPS
      "iree::builtins::ukernel::arch::arm_64::mmt4d_arm_64"
      "iree::builtins::ukernel::arch::arm_64::pack_arm_64"
      "iree::builtins::ukernel::arch::arm_64::query_tile_sizes_arm_64"
      "iree::builtins::ukernel::arch::arm_64::unpack_arm_64"
    )
  endif()
endif()  # IREE_UK_ENABLE_ARCH_SPECIFIC_CODE

set(IREE_UK_POINTER_SIZE "${CMAKE_SIZEOF_VOID_P}")

configure_file(config.h.in config.h)

iree_cc_library(
  NAME
    config
  HDRS
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

iree_cc_library(
  NAME
    ukernel_arch
  DEPS
    ${IREE_UK_ARCH_DEPS}
  PUBLIC
)
