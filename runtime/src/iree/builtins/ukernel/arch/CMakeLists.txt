# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if (NOT IREE_UK_ARCH_DEPS)
  set(IREE_UK_ARCH_DEPS "iree::builtins::ukernel::fallback")
endif()

iree_cc_library(
  NAME
    ukernel_arch
  DEPS
    ${IREE_UK_ARCH_DEPS}
  PUBLIC
)
