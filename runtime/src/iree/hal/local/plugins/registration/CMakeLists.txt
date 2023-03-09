# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_HAL_EXECUTABLE_PLUGIN_MODULES)
if(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF)
  list(APPEND IREE_HAL_EXECUTABLE_PLUGIN_MODULES iree::hal::local::plugins::embedded_elf_plugin)
endif()
if(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)
  list(APPEND IREE_HAL_EXECUTABLE_PLUGIN_MODULES iree::hal::local::plugins::system_library_plugin)
endif()

iree_cc_library(
  NAME
    registration
  HDRS
    "init.h"
  SRCS
    "init.c"
  DEPS
    iree::base
    iree::base::internal::flags
    iree::base::tracing
    iree::hal
    iree::hal::local::executable_plugin_manager
    ${IREE_HAL_EXECUTABLE_PLUGIN_EXTRA_DEPS}
    ${IREE_HAL_EXECUTABLE_PLUGIN_MODULES}
  PUBLIC
)
