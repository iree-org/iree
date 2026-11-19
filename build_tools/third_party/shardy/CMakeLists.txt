# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CMake build for Shardy dialect integration.
# Shardy (openxla/shardy) only has Bazel builds, so we provide minimal
# CMake support here for IREE integration.

message(STATUS "Building Shardy embedded in IREE")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(SHARDY_SOURCE_DIR "${IREE_SOURCE_DIR}/third_party/shardy")
set(SHARDY_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# Include directories for TableGen and compilation
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${SHARDY_SOURCE_DIR})
include_directories(${SHARDY_BINARY_DIR})

# Build common utilities
add_subdirectory(common)

# Build dialect
add_subdirectory(dialect)
