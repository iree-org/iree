# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wasm build infrastructure: optional freestanding libc and bundler tests.

if(IREE_WASM_BUILD_IREE_LIBC)
  add_subdirectory(libc)
endif()

if(IREE_BUILD_TESTS AND IREE_WASM_BUILD_IREE_LIBC)
  add_subdirectory(test)
endif()
