# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if (IREE_ARCH STREQUAL "arm_64")
  set(IREE_UK_ARCH_ARM_64 TRUE)
  list(APPEND IREE_UK_ARCH_DEPS
    "iree::builtins::ukernel::arch::arm_64"
  )
elseif (IREE_ARCH STREQUAL "x86_64")
  set(IREE_UK_ARCH_X86_64 TRUE)
  list(APPEND IREE_UK_ARCH_DEPS
    "iree::builtins::ukernel::arch::x86_64"
  )
endif()

iree_cc_library(
  NAME
    ukernel_arch
  DEPS
    ${IREE_UK_ARCH_DEPS}
  PUBLIC
)
