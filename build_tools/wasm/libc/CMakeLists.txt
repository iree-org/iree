# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Freestanding libc for wasm32.
#
# Provides C standard library headers and implementations for building the
# IREE runtime as a standalone wasm module. No system libc, no Emscripten —
# just enough to compile and link C code targeting wasm32-unknown-unknown.
#
# Headers are in include/ and are exposed via -isystem in the wasm32 toolchain
# file (build_tools/cmake/wasm32.cmake). The implementations are compiled here
# into iree_wasm_libc.

# JS implementation of the iree_syscall wasm imports.
iree_wasm_cc_library(
  NAME
    syscall_imports
  SRCS
    "src/syscall_imports.js"
  MODULE
    "iree_syscall"
  PUBLIC
)

add_library(iree_wasm_libc STATIC
  src/abort.c
  src/aligned_alloc.c
  src/ctype.c
  src/dlmalloc.c
  src/errno.c
  src/sbrk.c
  src/stdio.c
  src/stdlib.c
  src/strtol.c
  src/string.c
  src/unistd.c
)

target_include_directories(iree_wasm_libc PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# dlmalloc.c textually #includes "malloc.c" from third_party/dlmalloc/.
target_link_libraries(iree_wasm_libc PRIVATE dlmalloc::dlmalloc)
