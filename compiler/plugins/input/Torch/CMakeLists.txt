# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TORCH_MLIR_ROOT_DIR "${IREE_SOURCE_DIR}/third_party/torch-mlir")
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "")
set(IREE_COMPILER_TABLEGEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(torch-iree_compiler_defs INTERFACE)
target_include_directories(torch-iree_compiler_defs
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Configures all iree_cc_* targets to take this implicit dep,
# which provides common includes and copts for the tree.
set(IREE_IMPLICIT_DEFS_CC_DEPS torch-iree_compiler_defs)

add_subdirectory(torch-iree)
add_subdirectory(torch-mlir)
add_subdirectory(torch-mlir-dialects)
