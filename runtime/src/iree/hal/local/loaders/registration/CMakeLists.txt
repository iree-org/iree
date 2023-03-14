# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_HAL_EXECUTABLE_LOADER_MODULES)
if(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  list(APPEND IREE_HAL_EXECUTABLE_LOADER_MODULES iree::hal::local::loaders::embedded_elf_loader)
endif()
if(IREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY)
  list(APPEND IREE_HAL_EXECUTABLE_LOADER_MODULES iree::hal::local::loaders::system_library_loader)
endif()
if(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE)
  list(APPEND IREE_HAL_EXECUTABLE_LOADER_MODULES iree::hal::local::loaders::vmvx_module_loader)
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
    iree::hal::local
    ${IREE_HAL_EXECUTABLE_LOADER_EXTRA_DEPS}
    ${IREE_HAL_EXECUTABLE_LOADER_MODULES}
  PUBLIC
)
