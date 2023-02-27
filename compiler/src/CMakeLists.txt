# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    defs
  INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  PUBLIC
)

# Configures all iree_cc_* targets to take this implicit dep,
# which provides common includes and copts for the tree.
set(IREE_IMPLICIT_DEFS_CC_DEPS iree::compiler::src::defs)

# Force enable BUILD_SHARED_LIBS for the compiler if instructed.
set(_IREE_ORIG_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
if(IREE_COMPILER_BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(iree/compiler)

# Reset BUILD_SHARED_LIBS.
set(BUILD_SHARED_LIBS ${_IREE_ORIG_BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
