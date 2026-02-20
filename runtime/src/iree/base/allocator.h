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
// Checked arithmetic for allocation size calculations
//===----------------------------------------------------------------------===//

// Performs a checked addition of |a| and |b|, storing the result in
// |out_result|. Returns true if the addition succeeded without overflow, false
// if overflow occurred (|out_result| is undefined on overflow).
static inline bool iree_host_size_checked_add(iree_host_size_t a,
                                              iree_host_size_t b,
                                              iree_host_size_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_add_overflow)
  return !__builtin_add_overflow(a, b, out_result);
#else
  if (a > IREE_HOST_SIZE_MAX - b) return false;
  *out_result = a + b;
  return true;
#endif
}

// Performs a checked multiplication of |a| and |b|, storing the result in
// |out_result|. Returns true if the multiplication succeeded without overflow,
// false if overflow occurred (|out_result| is undefined on overflow).
static inline bool iree_host_size_checked_mul(iree_host_size_t a,
                                              iree_host_size_t b,
                                              iree_host_size_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_mul_overflow)
  return !__builtin_mul_overflow(a, b, out_result);
#else
  if (b != 0 && a > IREE_HOST_SIZE_MAX / b) return false;
  *out_result = a * b;
  return true;
#endif
}

// Performs checked computation of |base| + |count| * |element_size|, storing
// the result in |out_result|. Returns true if the computation succeeded without
// overflow, false if overflow occurred (|out_result| is undefined on overflow).
// This is the common pattern for computing allocation sizes with a header.
static inline bool iree_host_size_checked_mul_add(
    iree_host_size_t base, iree_host_size_t count,
    iree_host_size_t element_size, iree_host_size_t* out_result) {
  iree_host_size_t product = 0;
  if (!iree_host_size_checked_mul(count, element_size, &product)) return false;
  return iree_host_size_checked_add(base, product, out_result);
}

// Device size variants for remote device allocations.

static inline bool iree_device_size_checked_add(
    iree_device_size_t a, iree_device_size_t b,
    iree_device_size_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_add_overflow)
  return !__builtin_add_overflow(a, b, out_result);
#else
  if (a > IREE_DEVICE_SIZE_MAX - b) return false;
  *out_result = a + b;
  return true;
#endif
}

static inline bool iree_device_size_checked_mul(
    iree_device_size_t a, iree_device_size_t b,
    iree_device_size_t* out_result) {
#if IREE_HAVE_BUILTIN(__builtin_mul_overflow)
  return !__builtin_mul_overflow(a, b, out_result);
#else
  if (b != 0 && a > IREE_DEVICE_SIZE_MAX / b) return false;
  *out_result = a * b;
  return true;
#endif
}

static inline bool iree_device_size_checked_mul_add(
    iree_device_size_t base, iree_device_size_t count,
    iree_device_size_t element_size, iree_device_size_t* out_result) {
  iree_device_size_t product = 0;
  if (!iree_device_size_checked_mul(count, element_size, &product)) {
    return false;
  }
  return iree_device_size_checked_add(base, product, out_result);
}

// Aligns |value| up to |alignment| with overflow checking.
// |alignment| must be a power of two.
// Returns true if the alignment succeeded without overflow, false if overflow
// occurred (|out_aligned| is undefined on overflow).
static inline bool iree_host_size_checked_align(iree_host_size_t value,
                                                iree_host_size_t alignment,
                                                iree_host_size_t* out_aligned) {
  iree_host_size_t padded = 0;
  if (!iree_host_size_checked_add(value, alignment - 1, &padded)) {
    return false;
  }
  *out_aligned = padded & ~(alignment - 1);
  return true;
}

static inline bool iree_device_size_checked_align(
    iree_device_size_t value, iree_device_size_t alignment,
    iree_device_size_t* out_aligned) {
  iree_device_size_t padded = 0;
  if (!iree_device_size_checked_add(value, alignment - 1, &padded)) {
    return false;
  }
  *out_aligned = padded & ~(alignment - 1);
  return true;
}

//===----------------------------------------------------------------------===//
// Struct layout calculation
//===----------------------------------------------------------------------===//

// Descriptor for a single field in a struct layout calculation.
typedef struct iree_struct_field_t {
  // Element count per dimension ([0]*[1] = total).
  iree_host_size_t count[2];
  // Size of each element.
  iree_host_size_t element_size;
  // Required alignment (0 = none).
  iree_host_size_t alignment;
  // Output offset pointer (NULL to skip).
  iree_host_size_t* out_offset;
} iree_struct_field_t;

// Calculates the total allocation size for a struct with trailing fields.
// Each field can optionally capture its offset and specify alignment.
// All arithmetic is overflow-checked; returns IREE_STATUS_OUT_OF_RANGE on
// overflow.
//
// Example - struct with two trailing arrays:
//   iree_host_size_t total = 0, handles_offset = 0, fds_offset = 0;
//   IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
//       iree_sizeof_struct(*set), &total,
//       IREE_STRUCT_FIELD(capacity, iree_wait_handle_t, &handles_offset),
//       IREE_STRUCT_FIELD(capacity, struct pollfd, &fds_offset)));
//
// Example - header with cache-aligned data:
//   iree_host_size_t total = 0, data_offset = 0;
//   IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
//       0, &total,
//       IREE_STRUCT_FIELD_ALIGNED(1, header_t, iree_max_align_t, NULL),
//       IREE_STRUCT_FIELD_ALIGNED(size, uint8_t,
//       IREE_HAL_HEAP_BUFFER_ALIGNMENT,
//                                 &data_offset)));
//
// Example - 2D array (num_states rows of 256 columns):
//   iree_host_size_t total = 0, trans_offset = 0;
//   IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
//       sizeof(header_t), &total,
//       IREE_STRUCT_ARRAY_FIELD(num_states, 256, uint16_t, &trans_offset)));
IREE_ATTRIBUTE_ALWAYS_INLINE static inline iree_status_t
iree_struct_layout_calculate(iree_host_size_t base_size,
                             const iree_struct_field_t* fields,
                             iree_host_size_t field_count,
                             iree_host_size_t* out_total) {
  iree_host_size_t total = base_size;
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    const iree_struct_field_t* field = &fields[i];
    // Align offset if required.
    if (field->alignment > 0) {
      if (IREE_UNLIKELY(
              !iree_host_size_checked_align(total, field->alignment, &total))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "struct layout alignment overflow");
      }
    }
    // Record offset before adding this field.
    if (field->out_offset) {
      *field->out_offset = total;
    }
    // Checked multiply: count[0] * count[1] * element_size.
    iree_host_size_t element_count = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            field->count[0], field->count[1], &element_count))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "struct layout field count overflow");
    }
    iree_host_size_t field_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            element_count, field->element_size, &field_size))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "struct layout field size overflow");
    }
    // Checked add to running total.
    if (IREE_UNLIKELY(!iree_host_size_checked_add(total, field_size, &total))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "struct layout total size overflow");
    }
  }
  *out_total = total;
  return iree_ok_status();
}

// Field descriptor for an unaligned array.
#define IREE_STRUCT_FIELD(count_expr, type, out_offset_ptr) \
  { {(count_expr), 1}, sizeof(type), 0, (out_offset_ptr) }

// Field descriptor for an unaligned 2D array (count1 * count2 elements).
// Both count multiplications are overflow-checked.
#define IREE_STRUCT_ARRAY_FIELD(count1, count2, type, out_offset_ptr) \
  { {(count1), (count2)}, sizeof(type), 0, (out_offset_ptr) }

// Field descriptor for an aligned array.
#define IREE_STRUCT_FIELD_ALIGNED(count_expr, type, align, out_offset_ptr) \
  { {(count_expr), 1}, sizeof(type), (align), (out_offset_ptr) }

// Field descriptor for a flexible array member (FAM). FAMs are accessed via
// the struct member (e.g., foo->bar[]) so no offset is needed. The alignment
// ensures the FAM starts at an address suitable for its element type.
#define IREE_STRUCT_FIELD_FAM(count_expr, type) \
  { {(count_expr), 1}, sizeof(type), iree_alignof(type), NULL }

// Calculates struct layout using inline field descriptors.
// C++ version uses a lambda to create a local array (compound literals are a
// GCC extension in C++ and taking their address fails with GCC).
#ifdef __cplusplus
#define IREE_STRUCT_LAYOUT(base_size, out_total, ...)                      \
  [&]() -> iree_status_t {                                                 \
    const iree_struct_field_t fields[] = {__VA_ARGS__};                    \
    return iree_struct_layout_calculate(                                   \
        (base_size), fields, sizeof(fields) / sizeof(iree_struct_field_t), \
        (out_total));                                                      \
  }()
#else
#define IREE_STRUCT_LAYOUT(base_size, out_total, ...)                         \
  iree_struct_layout_calculate((base_size),                                   \
                               (const iree_struct_field_t[]){__VA_ARGS__},    \
                               sizeof((iree_struct_field_t[]){__VA_ARGS__}) / \
                                   sizeof(iree_struct_field_t),               \
                               (out_total))
#endif  // __cplusplus

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
// Safe to pass NULL (no-op).
IREE_API_EXPORT void iree_allocator_free(iree_allocator_t allocator, void* ptr);

//===----------------------------------------------------------------------===//
// Array allocation helpers with overflow checking
//===----------------------------------------------------------------------===//

// Allocates memory for |count| elements of |element_size| bytes each.
// The contents of the returned memory is guaranteed to be zeroed.
// Returns IREE_STATUS_OUT_OF_RANGE if the size calculation overflows.
IREE_API_EXPORT iree_status_t
iree_allocator_malloc_array(iree_allocator_t allocator, iree_host_size_t count,
                            iree_host_size_t element_size, void** out_ptr);

// Allocates memory for |count| elements of |element_size| bytes each.
// The content of the buffer returned is undefined.
// Returns IREE_STATUS_OUT_OF_RANGE if the size calculation overflows.
IREE_API_EXPORT iree_status_t iree_allocator_malloc_array_uninitialized(
    iree_allocator_t allocator, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr);

// Reallocates memory for |count| elements of |element_size| bytes each.
// Returns IREE_STATUS_OUT_OF_RANGE if the size calculation overflows.
IREE_API_EXPORT iree_status_t
iree_allocator_realloc_array(iree_allocator_t allocator, iree_host_size_t count,
                             iree_host_size_t element_size, void** inout_ptr);

// Allocates memory for a structure of |struct_size| bytes followed by
// |trailing_size| bytes of additional data. The combined allocation is
// aligned to iree_max_align_t.
// Returns IREE_STATUS_OUT_OF_RANGE if the size calculation overflows.
IREE_API_EXPORT iree_status_t iree_allocator_malloc_with_trailing(
    iree_allocator_t allocator, iree_host_size_t struct_size,
    iree_host_size_t trailing_size, void** out_ptr);

// Allocates memory for a structure of |struct_size| bytes followed by an
// array of |count| elements of |element_size| bytes each.
// Returns IREE_STATUS_OUT_OF_RANGE if the size calculation overflows.
IREE_API_EXPORT iree_status_t iree_allocator_malloc_struct_array(
    iree_allocator_t allocator, iree_host_size_t struct_size,
    iree_host_size_t count, iree_host_size_t element_size, void** out_ptr);

// Grows an array allocation using a 2x doubling strategy.
// The new capacity is max(|minimum_capacity|, |*inout_capacity| * 2).
// On success |*inout_capacity| is updated and |*inout_ptr| is reallocated.
// Returns IREE_STATUS_OUT_OF_RANGE if the capacity calculation overflows.
IREE_API_EXPORT iree_status_t iree_allocator_grow_array(
    iree_allocator_t allocator, iree_host_size_t minimum_capacity,
    iree_host_size_t element_size, iree_host_size_t* inout_capacity,
    void** inout_ptr);

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
