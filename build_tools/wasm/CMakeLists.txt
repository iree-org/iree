# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wasm build infrastructure: optional freestanding libc and bundler tests.

if(IREE_WASM_BUILD_IREE_LIBC)
  add_subdirectory(libc)
endif()

if(IREE_BUILD_TESTS)
  add_test(
    NAME build_tools_wasm_binary_bundler_test
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/wasm_binary_bundler_test.py"
  )
  iree_configure_test(build_tools_wasm_binary_bundler_test)
endif()

if(IREE_BUILD_TESTS AND
   (IREE_WASM_BUILD_IREE_LIBC OR CMAKE_SYSTEM_NAME STREQUAL "WASI"))
  add_subdirectory(test)
endif()
