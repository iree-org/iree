// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// dlmalloc configured for wasm32 freestanding.
//
// Doug Lea's malloc (v2.8.6, MIT-0 license) configured to use our sbrk()
// implementation (which wraps wasm memory.grow). No mmap, no locks (single-
// threaded), no system headers. Adds ~10KB to the wasm binary.
//
// All configuration is via #defines before including the upstream source.
// The upstream source is unmodified (third_party/dlmalloc/malloc.c).

// --- Configuration for wasm32 freestanding ---

// Use our sbrk() (in sbrk.c) as the memory source. No mmap.
#define HAVE_MMAP 0
#define HAVE_MREMAP 0
#define HAVE_MORECORE 1

// sbrk always returns contiguous memory (wasm linear memory is contiguous).
#define MORECORE_CONTIGUOUS 1

// No threading — single-threaded wasm.
#define USE_LOCKS 0

// No mspace support needed — IREE uses a single heap.
#define MSPACES 0
#define ONLY_MSPACES 0

// Standard 8-byte alignment for wasm32 (2 * sizeof(void*) = 8).
// This matches the natural alignment for double and int64_t.

// Error handling: set errno on failure. Our abort() calls the JS host.
#define MALLOC_FAILURE_ACTION errno = ENOMEM

// Disable debug checks in release builds (controlled by NDEBUG).
#ifndef NDEBUG
#define DEBUG 1
#else
#define DEBUG 0
#endif

// Disable footers — they add 4 bytes per allocation for checking which
// mspace an allocation came from. We only have one mspace.
#define FOOTERS 0

// No need for the dl_ prefix — we're the only malloc implementation.
// (USE_DL_PREFIX is intentionally NOT defined.)

// dlmalloc uses these from libc. Provide declarations so it compiles
// without system headers.
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memcpy, memset (wasm builtins via -mbulk-memory).

// Prevent dlmalloc from including system headers it doesn't need.
#define LACKS_SYS_PARAM_H
#define LACKS_SYS_MMAN_H
#define LACKS_SYS_TYPES_H
#define LACKS_STRINGS_H
#define LACKS_SCHED_H
#define LACKS_TIME_H
#define LACKS_UNISTD_H
#define LACKS_FCNTL_H

// sbrk declaration (implemented in sbrk.c).
// Uses ptrdiff_t to match dlmalloc's own declaration.
extern void* sbrk(ptrdiff_t increment);

// dlmalloc references abort() for fatal internal errors.
#include <stdlib.h>

// --- Include the upstream source ---
#include "malloc.c"  // NOLINT(build/include) - textual include of upstream dlmalloc
