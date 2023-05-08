# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Our inline assembly is of the GCC/Clang flavor. Enable it by default when it
# is expected to compile.
iree_select_compiler_opts(IREE_UK_ENABLE_INLINE_ASM_DEFAULT
  CLANG_OR_GCC
    ON
)

option(IREE_UK_ENABLE_INLINE_ASM 
  "Enable inline assembly in ukernels"
  ${IREE_UK_ENABLE_INLINE_ASM_DEFAULT}
)

option(IREE_UK_ENABLE_INTRINSICS_EVEN_WHEN_INLINE_ASM_AVAILABLE
  "Enable intrinsics code paths even when an inline assembly alternative is available. These may then be selected by a *_PREFER_INTRINSICS ukernel flag."
  OFF
)

if (IREE_ARCH STREQUAL "arm_64")
  set(IREE_UK_ARCH_ARM_64 TRUE)
  add_subdirectory(arm_64)
  list(APPEND IREE_UK_ARCH_DEPS
    "iree::builtins::ukernel::arch::arm_64"
  )
elseif (IREE_ARCH STREQUAL "x86_64")
  set(IREE_UK_ARCH_X86_64 TRUE)
  add_subdirectory(x86_64)
  list(APPEND IREE_UK_ARCH_DEPS
    "iree::builtins::ukernel::arch::x86_64"
  )
endif()

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
