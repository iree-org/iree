# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Shared library containing the MIPS custom matmul kernel.
#
# The kernel is exposed as an IREE executable plugin (implements
# iree_hal_executable_plugin_query) so that iree-run-module can load it via:
#   iree-run-module --executable_plugin=libmy_matmul_kernel.dylib ...
#
# The IREE runtime resolves the @my_matmul_kernel import from the dispatch
# executable through this plugin's resolve() function.

add_library(my_matmul_kernel SHARED
  my_matmul_kernel.c
)

target_include_directories(my_matmul_kernel
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    # For iree/hal/local/executable_plugin.h (standalone C header, no deps)
    ${PROJECT_SOURCE_DIR}/runtime/src
)

set_target_properties(my_matmul_kernel PROPERTIES
  C_VISIBILITY_PRESET default
  POSITION_INDEPENDENT_CODE ON
)
