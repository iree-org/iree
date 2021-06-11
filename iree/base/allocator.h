// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_ALLOCATOR_H_
#define IREE_BASE_ALLOCATOR_H_

#include <memory.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"
#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Returns the number of elements in an array as a compile-time constant, which
// can be used in defining new arrays. Fails at compile-time if |arr| is not a
// static array (such as if used on a pointer type).
//
// Example:
//  uint8_t kConstantArray[512];
//  assert(IREE_ARRAYSIZE(kConstantArray) == 512);
#define IREE_ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define iree_min(lhs, rhs) ((lhs) <= (rhs) ? (lhs) : (rhs))
#define iree_max(lhs, rhs) ((lhs) <= (rhs) ? (rhs) : (lhs))

// Returns true if any bit from |rhs| is set in |lhs|.
#define iree_any_bit_set(lhs, rhs) (((lhs) & (rhs)) != 0)
// Returns true iff all bits from |rhs| are set in |lhs|.
#define iree_all_bits_set(lhs, rhs) (((lhs) & (rhs)) == (rhs))

//===----------------------------------------------------------------------===//
// Byte buffers and memory utilities
//===----------------------------------------------------------------------===//

// A span of mutable bytes (ala std::span of uint8_t).
typedef struct iree_byte_span_t {
  uint8_t* data;
  iree_host_size_t data_length;
} iree_byte_span_t;

static inline iree_byte_span_t iree_make_byte_span(
    void* data, iree_host_size_t data_length) {
  iree_byte_span_t v = {(uint8_t*)data, data_length};
  return v;
}

// A span of constant bytes (ala std::span of const uint8_t).
typedef struct iree_const_byte_span_t {
  const uint8_t* data;
  iree_host_size_t data_length;
} iree_const_byte_span_t;

static inline iree_const_byte_span_t iree_make_const_byte_span(
    const void* data, iree_host_size_t data_length) {
  iree_const_byte_span_t v = {(const uint8_t*)data, data_length};
  return v;
}

//===----------------------------------------------------------------------===//
// Totally shady stack allocation
//===----------------------------------------------------------------------===//
// TODO(benvanik): remove our uses of this or make them more explicit.

#if defined(IREE_COMPILER_MSVC)
// The safe malloca that may fall back to heap in the case of stack overflows:
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/malloca?view=vs-2019
// Because that gets really annoying to deal with during error handling we just
// go for _alloca which may generate SEH exceptions if we blow the stack.
#include <malloc.h>
#define iree_alloca(sz) _alloca(sz)
#else
#include <alloca.h>
#define iree_alloca(sz) alloca(sz)
#endif  // IREE_COMPILER_MSVC

//===----------------------------------------------------------------------===//
// iree_allocator_t (std::allocator-like interface)
//===----------------------------------------------------------------------===//

// Defines how an allocation from an iree_allocator_t should be made.
enum iree_allocation_mode_bits_t {
  // The contents of the allocation *must* be zeroed by the allocator prior to
  // returning. Allocators may be able to elide the zeroing if they allocate
  // fresh pages from the system. It is always safe to zero contents if the
  // behavior of the allocator is not under our control.
  IREE_ALLOCATION_MODE_ZERO_CONTENTS = 1u << 0,
  // Tries to reuse an existing allocation provided via |out_ptr| if possible.
  // If the existing allocation is not reused then it is freed as if a call to
  // iree_allocator_free had been called on it. If the allocation fails then
  // the provided existing allocation is unmodified.
  //
  // This models the C realloc behavior.
  IREE_ALLOCATION_MODE_TRY_REUSE_EXISTING = 1u << 1,
};
typedef uint32_t iree_allocation_mode_t;

// TODO(benvanik): replace with a single method with the mode setting. This will
// reduce the overhead to just two pointers per allocator (from 3) and allow us
// to add more distinct behavior in the future. If we really wanted to stretch
// we could turn it into a pointer and require the allocator live somewhere
// (possibly in .text as a const static), but two pointers seems fine.
typedef iree_status_t(IREE_API_PTR* iree_allocator_alloc_fn_t)(
    void* self, iree_allocation_mode_t mode, iree_host_size_t byte_length,
    void** out_ptr);
typedef void(IREE_API_PTR* iree_allocator_free_fn_t)(void* self, void* ptr);

// An allocator for host-memory allocations.
// IREE will attempt to use this in place of the system malloc and free.
// Pass the iree_allocator_system() macro to use the system allocator.
typedef struct iree_allocator_t {
  // User-defined pointer passed to all functions.
  void* self;
  // Allocates |byte_length| of memory and stores the pointer in |out_ptr|.
  // Systems should align to 16 byte boundaries (or otherwise their natural
  // SIMD alignment). The runtime pools internally and small allocations
  // (usually) won't be made through this interface.
  iree_allocator_alloc_fn_t alloc;
  // Frees |ptr| from a previous alloc call.
  iree_allocator_free_fn_t free;
} iree_allocator_t;

// Allocates a block of |byte_length| bytes from the given allocator.
// The contents of the returned memory is guaranteed to be zeroed.
IREE_API_EXPORT iree_status_t iree_allocator_malloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr);

// Reallocates |out_ptr| to |byte_length| bytes with the given allocator.
// If the reallocation fails then the original |out_ptr| is unmodified.
IREE_API_EXPORT iree_status_t iree_allocator_realloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr);

// Duplicates the given byte block by allocating memory and copying it in.
IREE_API_EXPORT iree_status_t
iree_allocator_clone(iree_allocator_t allocator,
                     iree_const_byte_span_t source_bytes, void** out_ptr);

// Frees a previously-allocated block of memory to the given allocator.
IREE_API_EXPORT void iree_allocator_free(iree_allocator_t allocator, void* ptr);

// Allocates a block of |byte_length| bytes from the default system allocator.
IREE_API_EXPORT iree_status_t
iree_allocator_system_allocate(void* self, iree_allocation_mode_t mode,
                               iree_host_size_t byte_length, void** out_ptr);

// Frees a previously-allocated block of memory to the default system allocator.
IREE_API_EXPORT void iree_allocator_system_free(void* self, void* ptr);

// Allocates using the iree_allocator_malloc and iree_allocator_free methods.
// These will usually be backed by malloc and free.
static inline iree_allocator_t iree_allocator_system(void) {
  iree_allocator_t v = {NULL, iree_allocator_system_allocate,
                        iree_allocator_system_free};
  return v;
}

// Does not perform any allocation or deallocation; used to wrap objects that
// are owned by external code/live in read-only memory/etc.
static inline iree_allocator_t iree_allocator_null(void) {
  iree_allocator_t v = {NULL, NULL, NULL};
  return v;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_ALLOCATOR_H_
