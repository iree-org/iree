# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    defs
  INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}/..
  PUBLIC
)

# Configures all iree_cc_* targets to take this implicit dep,
# which provides common includes and copts for the tree.
# Must come after the defs target itself or else there will be a circular dep.
set(IREE_IMPLICIT_DEFS_CC_DEPS iree::defs)

iree_add_all_subdirs()
