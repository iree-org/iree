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

#if IREE_STATISTICS_ENABLE
// Evalutes the expression code only if statistics are enabled.
//
// Example:
//  struct {
//    IREE_STATISTICS(uint32_t stats_only_value);
//  } my_object;
//  IREE_STATISTICS(my_object.stats_only_value = 5);
//  IREE_STATISTICS({
//    my_object.stats_only_value = 5;
//  });
#define IREE_STATISTICS(expr) expr
#else
#define IREE_STATISTICS(expr)
#endif  // IREE_STATISTICS_ENABLE

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

static inline iree_byte_span_t iree_byte_span_empty() {
  iree_byte_span_t v = {NULL, 0};
  return v;
}

static inline bool iree_byte_span_is_empty(iree_byte_span_t span) {
  return span.data == NULL || span.data_length == 0;
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

static inline iree_const_byte_span_t iree_const_byte_span_empty() {
  iree_const_byte_span_t v = {NULL, 0};
  return v;
}

static inline bool iree_const_byte_span_is_empty(iree_const_byte_span_t span) {
  return span.data == NULL || span.data_length == 0;
}

static inline iree_const_byte_span_t iree_const_cast_byte_span(
    iree_byte_span_t span) {
  return iree_make_const_byte_span(span.data, span.data_length);
}

static inline iree_byte_span_t iree_cast_const_byte_span(
    iree_const_byte_span_t span) {
  return iree_make_byte_span((uint8_t*)span.data, span.data_length);
}

// Copies |size| bytes from |src| to |dst| without polluting the cache with
// |dst| lines. Used when streaming data that will not be read again.
static inline void iree_memcpy_stream_dst(void* IREE_RESTRICT dst,
                                          const void* IREE_RESTRICT src,
                                          iree_host_size_t size) {
  // TODO(benvanik): implement a proper non-temporal copy. This will be
  // architecture-specific and may have compiler-specific paths in order to emit
  // the proper instructions. On x64 this should be using MOVNTDQ (or something
  // in that family).
  memcpy(dst, src, size);
}

//===----------------------------------------------------------------------===//
// Totally shady stack allocation
//===----------------------------------------------------------------------===//
// TODO(benvanik): remove our uses of this or make them more explicit.

#if defined(IREE_PLATFORM_WINDOWS)
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
  IREE_ALLOCATOR_COMMAND_CALLOC = 1,

  // Tries to resize an allocation provided via |inout_ptr|, if possible.
  // If the existing allocation is not reused then it is freed as if a call to
  // iree_allocator_free had been called on it. If the allocation fails then
  // the provided existing allocation is unmodified. Only pointers previously
  // received from the iree_allocator_t are valid.
  //
  // iree_allocator_ctl_fn_t:
  //   params: iree_allocator_alloc_params_t
  //   inout_ptr: pointer of existing allocation; updated to realloced pointer
  IREE_ALLOCATOR_COMMAND_REALLOC = 2,

  // Frees the memory pointed to by |inout_ptr|.
  //
  // iree_allocator_ctl_fn_t:
  //   params: unused
  //   inout_ptr: pointer to free
  IREE_ALLOCATOR_COMMAND_FREE = 3,

  // TODO(benvanik): add optional IREE_ALLOCATOR_COMMAND_BIND like mbind:
  // https://man7.org/linux/man-pages/man2/mbind.2.html
  // This would take a pointer/length and a NUMA node ID to bind the memory to.
  // We may want flags for controlling whether this is a new allocation getting
  // bound or an existing one that is migrating to use MPOL_MF_MOVE.
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
// is defined by each operation but is generally a pointer to the pointer to
// set to the newly allocated memory or a pointer to the pointer to free.
typedef iree_status_t(IREE_API_PTR* iree_allocator_ctl_fn_t)(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_ptr);

// An allocator for host-memory allocations.
// IREE will attempt to use this in place of the system malloc and free.
// Pass the iree_allocator_system() macro to use the system allocator.
typedef struct iree_allocator_t {
  // Control function data.
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
// TODO(benvanik): make them zeros; we should have an _uninitialized if needed.
IREE_API_EXPORT iree_status_t iree_allocator_realloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** inout_ptr);

// Duplicates the given byte block by allocating memory and copying it in.
IREE_API_EXPORT iree_status_t
iree_allocator_clone(iree_allocator_t allocator,
                     iree_const_byte_span_t source_bytes, void** out_ptr);

// Frees a previously-allocated block of memory to the given allocator.
IREE_API_EXPORT void iree_allocator_free(iree_allocator_t allocator, void* ptr);

//===----------------------------------------------------------------------===//
// Built-in iree_allocator_t implementations
//===----------------------------------------------------------------------===//

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

typedef struct {
  iree_host_size_t capacity;
  iree_host_size_t length;
  iree_host_size_t head_size;
  uint8_t* buffer;
} iree_allocator_inline_storage_t;

// Stack storage for an inline arena-style allocator.
//
// Usage:
//  IREE_ALLOCATOR_INLINE_STORAGE(inline_storage, 2048);
//  something_allocating(iree_allocator_inline_arena(&inline_storage.header));
#define IREE_ALLOCATOR_INLINE_STORAGE(var, storage_capacity) \
  struct {                                                   \
    iree_allocator_inline_storage_t header;                  \
    uint8_t data[storage_capacity];                          \
  } var = {                                                  \
      .header =                                              \
          {                                                  \
              .capacity = sizeof((var).data),                \
              .length = 0,                                   \
              .buffer = &(var).data[0],                      \
          },                                                 \
  };

// Inline arena allocator controller used by iree_allocator_inline_arena.
IREE_API_EXPORT iree_status_t
iree_allocator_inline_arena_ctl(void* self, iree_allocator_command_t command,
                                const void* params, void** inout_ptr);

// Allocates with arena semantics within the given fixed-size |storage|.
// Frees are ignored and all allocations will fail once the allocated length
// exceeds the capacity. A special case for reallocations of the entire
// outstanding memory is supported to allow the arena to be implicitly reset.
static inline iree_allocator_t iree_allocator_inline_arena(
    iree_allocator_inline_storage_t* storage) {
  iree_allocator_t v = {storage, iree_allocator_inline_arena_ctl};
  return v;
}

//===----------------------------------------------------------------------===//
// System allocator support (builtin or user-provided)
//===----------------------------------------------------------------------===//

#if defined(IREE_ALLOCATOR_SYSTEM_CTL)

// System allocator provided by the user as part of build-time configuration.
// The implementation need only be linked into final executable binaries.
IREE_API_EXPORT iree_status_t
IREE_ALLOCATOR_SYSTEM_CTL(void* self, iree_allocator_command_t command,
                          const void* params, void** inout_ptr);

#if defined(IREE_ALLOCATOR_SYSTEM_SELF)
// Optional self for the default allocator.
// Must be defined but may be NULL.
IREE_API_EXPORT void* IREE_ALLOCATOR_SYSTEM_SELF;
#else
#define IREE_ALLOCATOR_SYSTEM_SELF NULL
#endif  // IREE_ALLOCATOR_SYSTEM_SELF

// System allocator provided by the user as part of build-time configuration
// (or a fallback of `malloc` and `free`).
//
// Specified by defining `IREE_ALLOCATOR_SYSTEM_CTL`, an implementation of the
// allocator control function (see `iree_allocator_ctl_fn_t`). An optional
// `IREE_ALLOCATOR_SYSTEM_SELF` global `void*` variable can be defined if the
// allocator requires state and otherwise `NULL` will be passed as the `self`
// parameter to the control function.
static inline iree_allocator_t iree_allocator_system(void) {
  iree_allocator_t v = {
      IREE_ALLOCATOR_SYSTEM_SELF,
      IREE_ALLOCATOR_SYSTEM_CTL,
  };
  return v;
}

#endif  // IREE_ALLOCATOR_SYSTEM_CTL

//===----------------------------------------------------------------------===//
// Aligned allocations via iree_allocator_t
//===----------------------------------------------------------------------===//

// Allocates memory of size |byte_length| where the byte starting at |offset|
// has a minimum alignment of |min_alignment|. In many cases |offset| can be 0.
//
// The |offset| can be used to ensure the alignment-sensitive portion of a
// combined allocation is aligned while any prefix metadata has system
// alignment. For example:
//   typedef struct {
//     uint32_t some_metadata;
//     uint8_t data[];
//   } buffer_t;
//   buffer_t* buffer = NULL;
//   iree_allocator_malloc_aligned(allocator, sizeof(buffer_t) + length,
//                                 4096, offsetof(buffer_t, data), &buffer);
//   // `buffer` has system alignment, but the `data` will be aligned on at
//   // least a 4096 boundary.
//
// The contents of the returned memory is guaranteed to be zeroed.
IREE_API_EXPORT iree_status_t iree_allocator_malloc_aligned(
    iree_allocator_t allocator, iree_host_size_t byte_length,
    iree_host_size_t min_alignment, iree_host_size_t offset, void** out_ptr);

// Reallocates memory to |byte_length|, growing or shrinking as needed.
// Only valid on memory allocated with iree_allocator_malloc_aligned.
// The newly reallocated memory will have the byte at |offset| aligned to at
// least |min_alignment|.
IREE_API_EXPORT iree_status_t iree_allocator_realloc_aligned(
    iree_allocator_t allocator, iree_host_size_t byte_length,
    iree_host_size_t min_alignment, iree_host_size_t offset, void** inout_ptr);

// Frees a |ptr| previously returned from iree_allocator_malloc_aligned.
IREE_API_EXPORT void iree_allocator_free_aligned(iree_allocator_t allocator,
                                                 void* ptr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_ALLOCATOR_H_
