// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// sbrk() for wasm32 — thin wrapper over the memory.grow instruction.
//
// The wasm linear memory is divided into three regions by wasm-ld:
//
//   [0, __data_end)           Static data (.data, .rodata, .bss segments).
//   [__data_end, __heap_base) Stack (when linked with --stack-first, the
//                             stack is placed between data and heap).
//   [__heap_base, ...)        Heap (managed by dlmalloc via this sbrk).
//
// memory.grow operates in 64KB page granularity and returns the previous
// memory size in pages on success, or (size_t)-1 on failure. It never traps —
// failure is indicated by the return value. Growth is one-way: memory cannot
// be returned to the host.
//
// This sbrk is initialized lazily from __heap_base on first call.

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

// Provided by wasm-ld. First byte after static data + stack.
extern unsigned char __heap_base;

#define WASM_PAGE_SIZE 65536

static unsigned char* heap_end = NULL;

void* sbrk(ptrdiff_t increment) {
  if (!heap_end) {
    heap_end = &__heap_base;
  }

  uintptr_t old_end = (uintptr_t)heap_end;
  uintptr_t heap_base = (uintptr_t)&__heap_base;

  if (increment > 0) {
    uintptr_t uincrement = (uintptr_t)increment;

    // Check for 32-bit address space overflow before forming the pointer.
    if (uincrement > UINTPTR_MAX - old_end) {
      errno = ENOMEM;
      return (void*)-1;
    }
    uintptr_t new_end = old_end + uincrement;

    // Compute how many new pages are needed beyond what's already mapped.
    // Use 64-bit arithmetic to avoid overflow at the 4GB boundary (65536
    // pages * 65536 bytes = 2^32, which wraps to 0 in 32-bit math).
    uint64_t current_memory_bytes =
        (uint64_t)__builtin_wasm_memory_size(0) * WASM_PAGE_SIZE;
    if (new_end > current_memory_bytes) {
      uint64_t deficit = (uint64_t)new_end - current_memory_bytes;
      size_t pages_needed =
          (size_t)((deficit + WASM_PAGE_SIZE - 1) / WASM_PAGE_SIZE);
      size_t result = __builtin_wasm_memory_grow(0, pages_needed);
      if (result == (size_t)-1) {
        errno = ENOMEM;
        return (void*)-1;
      }
    }
  } else if (increment < 0) {
    uintptr_t magnitude = (uintptr_t)(-(uintptr_t)increment);

    // Negative sbrk: just move the pointer back. Memory pages are never
    // returned (wasm doesn't support shrinking linear memory). dlmalloc
    // uses negative sbrk to "return" memory, but it's a no-op for us
    // beyond bookkeeping.
    if (magnitude > old_end - heap_base) {
      errno = ENOMEM;
      return (void*)-1;
    }
  }

  heap_end = (unsigned char*)(old_end + (uintptr_t)increment);
  return (void*)old_end;
}
