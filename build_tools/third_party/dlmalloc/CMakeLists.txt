# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Doug Lea's malloc (dlmalloc) v2.8.6, MIT-0 license.
# https://gee.cs.oswego.edu/dl/html/malloc.html
#
# The upstream source (malloc.c) is not compiled directly - it is textually
# #included by our configuration wrapper (build_tools/wasm/libc/src/dlmalloc.c).
# This target provides the include path so the wrapper can find it.

set(DLMALLOC_ROOT "${IREE_ROOT_DIR}/third_party/dlmalloc")

add_library(dlmalloc INTERFACE)
target_include_directories(dlmalloc INTERFACE "${DLMALLOC_ROOT}")
add_library(dlmalloc::dlmalloc ALIAS dlmalloc)
