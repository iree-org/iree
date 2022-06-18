# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(ExternalProject)

# Since we are only using NCCL via dynamic linking, we just need some of the
# headers. These are built as part of the NCCL Makefiles (they include some
# substitutions), and we only generate those.
find_program(MAKE_EXE NAMES gmake nmake make)
ExternalProject_add(iree_nccl_headers_import
  SOURCE_DIR "${IREE_SOURCE_DIR}/third_party/nccl/src"
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_COMMAND "${MAKE_EXE}"
    "${CMAKE_CURRENT_BINARY_DIR}/include/nccl.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/nccl_net.h"
    "BUILDDIR=${CMAKE_CURRENT_BINARY_DIR}"
)

add_library(iree_nccl_headers INTERFACE)
add_dependencies(iree_nccl_headers iree_nccl_headers_import)
target_include_directories(iree_nccl_headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)

target_compile_definitions(iree_nccl_headers
  INTERFACE
    IREE_HAS_NCCL_HEADERS
)
