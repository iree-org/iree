// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C11 aligned_alloc for wasm32.
//
// dlmalloc provides memalign() but not the C11 aligned_alloc(). They have
// the same underlying behavior (memalign is actually more permissive about
// the size argument). This thin wrapper satisfies the C11 requirement.

#include <stdlib.h>

// Provided by dlmalloc.
extern void* memalign(size_t alignment, size_t size);

void* aligned_alloc(size_t alignment, size_t size) {
  return memalign(alignment, size);
}
