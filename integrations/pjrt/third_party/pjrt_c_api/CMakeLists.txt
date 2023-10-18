# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree_pjrt_deps")

add_library(iree_pjrt_c_api_compiler_defs INTERFACE)
target_include_directories(iree_pjrt_c_api_compiler_defs
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Configures all iree_cc_* targets to take this implicit dep,
# which provides common includes and copts for the tree.
set(IREE_IMPLICIT_DEFS_CC_DEPS iree_pjrt_c_api_compiler_defs)


iree_cc_library(
  NAME
    headers
  HDRS
    "xla/pjrt/c/pjrt_c_api.h"
  PUBLIC
)
