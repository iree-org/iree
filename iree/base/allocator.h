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

#include "iree/base/alignment.h"
#include "iree/base/attributes.h"
#include "iree/base/config.h"
#include "iree/base/status.h"
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
// C11 aligned_alloc compatibility shim
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/aligned-malloc
#define iree_aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define iree_aligned_free(p) _aligned_free(p)
#elif defined(_ISOC11_SOURCE)
// https://en.cppreference.com/w/c/memory/aligned_alloc
#define iree_aligned_alloc(alignment, size) aligned_alloc(alignment, size)
#define iree_aligned_free(p) free(p)
#elif _POSIX_C_SOURCE >= 200112L
// https://pubs.opengroup.org/onlinepubs/9699919799/functions/posix_memalign.html
static inline void* iree_aligned_alloc(size_t alignment, size_t size) {
  void* ptr = NULL;
  return posix_memalign(&ptr, alignment, size) == 0 ? ptr : NULL;
}
#define iree_aligned_free(p) free(p)
#else
// Emulates alignment with normal malloc. We overallocate by at least the
// alignment + the size of a pointer, store the base pointer at p[-1], and
// return the aligned pointer. This lets us easily get the base pointer in free
// to pass back to the system.
static inline void* iree_aligned_alloc(size_t alignment, size_t size) {
  void* base_ptr = malloc(size + alignment + sizeof(uintptr_t));
  if (!base_ptr) return NULL;
  uintptr_t* aligned_ptr = (uintptr_t*)iree_host_align(
      (uintptr_t)base_ptr + sizeof(uintptr_t), alignment);
  aligned_ptr[-1] = (uintptr_t)base_ptr;
  return aligned_ptr;
}
static inline void iree_aligned_free(void* p) {
  if (IREE_UNLIKELY(!p)) return;
  uintptr_t* aligned_ptr = (uintptr_t*)p;
  void* base_ptr = (void*)aligned_ptr[-1];
  free(base_ptr);
}
#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// iree_allocator_t (std::allocator-like interface)
//===----------------------------------------------------------------------===//

// Controls the behavior of an iree_allocator_ctl_fn_t callback function.
typedef enum iree_allocator_command_e {
  // Allocates |byte_length| of memory and stores the pointer in |inout_ptr|.
  // Systems should align to 16 byte boundaries (or otherwise their natural
  // SIMD alignment). The runtime pools internally and small allocations
  // (usually) won't be made through this interface.
  //
  // iree_allocator_ctl_fn_t:
  //   params: iree_allocator_alloc_params_t
  //   inout_ptr: set to allocated pointer
  IREE_ALLOCATOR_COMMAND_MALLOC = 0,

  // As with IREE_ALLOCATOR_COMMAND_MALLOC but zeros the memory.
  //
  // The contents of the allocation *must* be zeroed by the allocator prior to
  // returning. Allocators may be able to elide the zeroing if they allocate
  // fresh pages from the system. It is always safe to zero contents if the
  // behavior of the allocator is not under our control.
  //
  // iree_allocator_ctl_fn_t:
  //   params: iree_allocator_alloc_params_t
  //   inout_ptr: set to allocated pointer
  IREE_ALLOCATOR_COMMAND_CALLOC,

  // Tries to resize an allocation provided via |inout_ptr|, if possible.
  // If the existing allocation is not reused then it is freed as if a call to
  // iree_allocator_free had been called on it. If the allocation fails then
  // the provided existing allocation is unmodified. Only pointers previously
  // received from the iree_allocator_t are valid.
  //
  // iree_allocator_ctl_fn_t:
  //   params: iree_allocator_alloc_params_t
  //   inout_ptr: pointer of existing allocation; updated to realloced pointer
  IREE_ALLOCATOR_COMMAND_REALLOC,

  // Frees the memory pointed to by |inout_ptr|.
  //
  // iree_allocator_ctl_fn_t:
  //   params: unused
  //   inout_ptr: pointer to free
  IREE_ALLOCATOR_COMMAND_FREE,
} iree_allocator_command_t;

// Parameters for various allocation commands.
typedef struct iree_allocator_alloc_params_t {
  // Minimum size, in bytes, of the allocation. The underlying allocator may
  // pad the length out if needed.
  iree_host_size_t byte_length;
} iree_allocator_alloc_params_t;

// Function pointer for an iree_allocator_t control function.
// |command| provides the operation to perform. Optionally some commands may use
// |params| to pass additional operation-specific parameters. |inout_ptr| usage
// is defined by each operation but is general a pointer to the pointer to
// set to the newly allocated memory or a pointer to the pointer to free.
typedef iree_status_t(IREE_API_PTR* iree_allocator_ctl_fn_t)(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_ptr);

// An allocator for host-memory allocations.
// IREE will attempt to use this in place of the system malloc and free.
// Pass the iree_allocator_system() macro to use the system allocator.
typedef struct iree_allocator_t {
  // User-defined pointer passed to all functions.
  void* self;
  // ioctl-style control function servicing all allocator-related commands.
  // See iree_allocator_command_t for more information.
  iree_allocator_ctl_fn_t ctl;
} iree_allocator_t;

// Allocates a block of |byte_length| bytes from the given allocator.
// The contents of the returned memory is guaranteed to be zeroed.
IREE_API_EXPORT iree_status_t iree_allocator_malloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr);

// Allocates a block of |byte_length| bytes from the given allocator.
// The content of the buffer returned is undefined: it may be zeros, a
// debug-fill pattern, or random memory from elsewhere in the process.
// Only use this when immediately overwriting all memory.
IREE_API_EXPORT iree_status_t iree_allocator_malloc_uninitialized(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr);

// Reallocates |inout_ptr| to |byte_length| bytes with the given allocator.
// If the reallocation fails then the original |inout_ptr| is unmodified.
//
// WARNING: when extending the newly allocated bytes are undefined.
IREE_API_EXPORT iree_status_t iree_allocator_realloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** inout_ptr);

// Duplicates the given byte block by allocating memory and copying it in.
IREE_API_EXPORT iree_status_t
iree_allocator_clone(iree_allocator_t allocator,
                     iree_const_byte_span_t source_bytes, void** out_ptr);

// Frees a previously-allocated block of memory to the given allocator.
IREE_API_EXPORT void iree_allocator_free(iree_allocator_t allocator, void* ptr);

// Default C allocator controller using malloc/free.
IREE_API_EXPORT iree_status_t
iree_allocator_system_ctl(void* self, iree_allocator_command_t command,
                          const void* params, void** inout_ptr);

// Allocates using the iree_allocator_malloc and iree_allocator_free methods.
// These will usually be backed by malloc and free.
static inline iree_allocator_t iree_allocator_system(void) {
  iree_allocator_t v = {NULL, iree_allocator_system_ctl};
  return v;
}

// Does not perform any allocation or deallocation; used to wrap objects that
// are owned by external code/live in read-only memory/etc.
static inline iree_allocator_t iree_allocator_null(void) {
  iree_allocator_t v = {NULL, NULL};
  return v;
}

// Returns true if the allocator is `iree_allocator_null()`.
static inline bool iree_allocator_is_null(iree_allocator_t allocator) {
  return allocator.ctl == NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_ALLOCATOR_H_
