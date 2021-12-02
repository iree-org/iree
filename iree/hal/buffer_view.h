// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_BUFFER_VIEW_H_
#define IREE_HAL_BUFFER_VIEW_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// NOTE: these values must be in sync with
//    iree/compiler/Dialect/HAL/IR/HALTypes.cpp

enum iree_hal_numerical_type_bits_t {
  // Opaque or unknown - bytes cannot be interpreted. Indexing is still allowed
  // so long as the bit width of the elements is known.
  IREE_HAL_NUMERICAL_TYPE_UNKNOWN = 0x00u,

  // Signless integer-like.
  IREE_HAL_NUMERICAL_TYPE_INTEGER = 0x10u,
  // Signed integer.
  IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED =
      IREE_HAL_NUMERICAL_TYPE_INTEGER | 0x01u,
  // Unsigned integer.
  IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED =
      IREE_HAL_NUMERICAL_TYPE_INTEGER | 0x02u,

  // Float-like.
  IREE_HAL_NUMERICAL_TYPE_FLOAT = 0x20,
  // IEEE754-compatible floating point semantics.
  IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE = IREE_HAL_NUMERICAL_TYPE_FLOAT | 0x01u,
  // 'Brain' floating point semantics (currently only bf16).
  IREE_HAL_NUMERICAL_TYPE_FLOAT_BRAIN = IREE_HAL_NUMERICAL_TYPE_FLOAT | 0x02u,
};
typedef uint8_t iree_hal_numerical_type_t;

#define IREE_HAL_ELEMENT_TYPE_VALUE(numerical_type, bit_count) \
  (((uint32_t)(numerical_type) << 24) | (uint32_t)(bit_count))

// Composes an iree_hal_element_type_t value with the given attributes.
#define iree_hal_make_element_type(numerical_type, bit_count) \
  (iree_hal_element_type_t)(                                  \
      IREE_HAL_ELEMENT_TYPE_VALUE(numerical_type, bit_count))

// Returns the numerical type of the element, if known and not opaque.
#define iree_hal_element_numerical_type(element_type) \
  (iree_hal_numerical_type_t)((uint32_t)(element_type) >> 24)

// Returns true if |element_type| is opaque and cannot be interpreted.
#define iree_hal_element_numerical_type_is_opaque(element_type) \
  (iree_hal_element_numerical_type(element_type) ==             \
   IREE_HAL_NUMERICAL_TYPE_UNKNOWN)

// Returns true if |element_type| is an integer of some width and semantics.
#define iree_hal_element_numerical_type_is_integer(element_type)   \
  iree_all_bits_set(iree_hal_element_numerical_type(element_type), \
                    IREE_HAL_NUMERICAL_TYPE_INTEGER)

// Returns true if |element_type| is a float of some width and semantics.
#define iree_hal_element_numerical_type_is_float(element_type)     \
  iree_all_bits_set(iree_hal_element_numerical_type(element_type), \
                    IREE_HAL_NUMERICAL_TYPE_FLOAT)

// Returns the bit width of each element.
#define iree_hal_element_bit_count(element_type) (size_t)((element_type)&0xFF)

// Returns true if the element is byte-aligned.
// Sub-byte aligned types such as i4 require user handling of the packing.
#define iree_hal_element_is_byte_aligned(element_type) \
  (iree_hal_element_bit_count(element_type) % 8 == 0)

// Returns the number of bytes each |element_type| consumes in memory.
// This is only valid when the encoding type is dense as sub-byte bit widths
// may be packed in various forms (for example, i4 may be stored as nibbles
// where each byte in memory contains two elements).
#define iree_hal_element_dense_byte_count(element_type) \
  ((iree_hal_element_bit_count(element_type) + 8 - 1) / 8)

// Returns true if the given |element_type| represents an integer of exactly
// |bit_width|. This ignores the signedness of the integer type.
#define iree_hal_element_type_is_integer(element_type, bit_width) \
  (iree_hal_element_numerical_type_is_integer(element_type) &&    \
   iree_hal_element_bit_count(element_type) == (bit_width))

// Defines the element type of a buffer in a standard format.
//
// Composed as a 32-bit bitfield to allow for opaque data types. Use
// iree_hal_make_element_type to make a bitfield with the appropriate ordering.
//
//   MSB ----------------------------------------------- LSB
//   [numerical type] [reserved] [reserved] [number of bits]
//
// clang-format off
enum iree_hal_element_types_t {
  IREE_HAL_ELEMENT_TYPE_NONE             = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,             0),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_8         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,             8),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_16        = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,            16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_32        = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,            32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_64        = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,            64),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_INT_4            = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER,             4),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_4           = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,      4),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_4           = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,    4),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_INT_8            = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER,             8),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_8           = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,      8),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_8           = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,    8),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_INT_16           = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER,            16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_16          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,     16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_16          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,   16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_INT_32           = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER,            32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_32          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,     32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_32          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,   32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_INT_64           = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER,            64),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_64          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,     64),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_64          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,   64),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_FLOAT_16         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE,         16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_FLOAT_32         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE,         32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_FLOAT_64         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE,         64),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_BFLOAT_16        = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_BRAIN,        16),  // NOLINT
};
typedef uint32_t iree_hal_element_type_t;
// clang-format on

// Defines the encoding type of a buffer when known.
enum iree_hal_encoding_types_t {
  // Encoding is unknown or unspecified. Generic interpretation of the buffer
  // contents is not possible.
  IREE_HAL_ENCODING_TYPE_OPAQUE = 0,
  // Encoding is a densely-packed numpy/C-style row-major format.
  // All elements are contiguous in memory.
  IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR = 1,
  // TODO(#6762): sparse encodings we care about (_SPARSE_CSR)
  // We will likely want to make this a bitfield like the element type is that
  // we can more easily distinguish between encoding types that we can use for
  // certain operations; for example, size calculations on a DENSE_ROW_MAJOR
  // and DENSE_COLUMN_MAJOR would be easier to perform if we had a bit to test
  // for whether it's dense.
};
typedef uint32_t iree_hal_encoding_type_t;

// A dimension within a shape.
typedef int32_t iree_hal_dim_t;

//===----------------------------------------------------------------------===//
// Buffer view math
//===----------------------------------------------------------------------===//

// Calculates the allocation size of a buffer view.
IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_size(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_device_size_t* out_allocation_size);

// Calculates a byte offset into a buffer at the given indices.
// Only works with densely-packed representations.
IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_offset(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, const iree_hal_dim_t* indices,
    size_t indices_count, iree_device_size_t* out_offset);

// Calculates a byte range into a buffer of the given contiguous range.
// Only works with densely-packed representations.
IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_range(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, const iree_hal_dim_t* start_indices,
    iree_host_size_t indices_count, const iree_hal_dim_t* lengths,
    iree_host_size_t lengths_count, iree_device_size_t* out_start_offset,
    iree_device_size_t* out_length);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

// A shaped and typed view into a storage buffer.
// This is the closest thing to a "tensor" we have, and it's purely used to ease
// application code and not treated special internally by IREE. They are
// effectively just `tuple(shape, type, buffer)`, and if the application is
// already tracking this information in its own structures this entire type can
// be ignored.
typedef struct iree_hal_buffer_view_t iree_hal_buffer_view_t;

// Creates a buffer view with the given |buffer|.
// |out_buffer_view| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_create(
    iree_hal_buffer_t* buffer, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_hal_buffer_view_t** out_buffer_view);

// Allocates a buffer from |allocator| and wraps it in a buffer view.
// This is equivalent to:
//   1. iree_hal_buffer_compute_view_size
//   2. iree_hal_allocator_allocate_buffer
//   3. iree_hal_buffer_view_create
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_allocate_buffer(
    iree_hal_allocator_t* allocator, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_view_t** out_buffer_view);

// Clones a host buffer using |allocator| and wraps it in a buffer view.
// This is equivalent to:
//   1. iree_hal_allocator_allocate_buffer
//   2. iree_hal_buffer_write_data
//   3. iree_hal_buffer_view_create
//
// Always prefer allocating a device buffer and populating it in place.
// If cloning multiple buffers it is better to use iree_hal_command_buffer_ts to
// batch up the memory transfer operations.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_clone_heap_buffer(
    iree_hal_allocator_t* allocator, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_const_byte_span_t data,
    iree_hal_buffer_view_t** out_buffer_view);

// Imports a host buffer using |allocator| and wraps it in a buffer view.
// This is equivalent to:
//   1. iree_hal_allocator_wrap_buffer
//   2. iree_hal_buffer_view_create
//
// NOTE: not all buffers can be imported and not all allocators support
// importing. See iree_hal_allocator_wrap_buffer for more information.
// Fails if the buffer cannot be imported.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_wrap_heap_buffer(
    iree_hal_allocator_t* allocator, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_view_t** out_buffer_view);

// Tries to import a host buffer using |allocator| and wrap it in a buffer view.
// If the buffer cannot be imported then a new buffer will be allocated and the
// source data will be copied into it.
// This is equivalent to:
//   if iree_hal_allocator_query_buffer_compatibility ok:
//     1. iree_hal_allocator_wrap_buffer
//     2. iree_hal_buffer_view_create
//   else:
//     1. iree_hal_allocator_allocate_buffer
//     2. iree_hal_buffer_write_data
//     3. iree_hal_buffer_view_create
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_wrap_or_clone_heap_buffer(
    iree_hal_allocator_t* allocator, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_view_t** out_buffer_view);

// Retains the given |buffer_view| for the caller.
IREE_API_EXPORT void iree_hal_buffer_view_retain(
    iree_hal_buffer_view_t* buffer_view);

// Releases the given |buffer_view| from the caller.
IREE_API_EXPORT void iree_hal_buffer_view_release(
    iree_hal_buffer_view_t* buffer_view);

// Returns the buffer underlying the buffer view.
// The caller must retain the returned buffer if they want to continue using it.
//
// NOTE: the returned buffer length will almost always be larger than the valid
// bytes representing this buffer view due to padding. Always query the actual
// valid length with iree_hal_buffer_view_byte_length instead of assuming the
// buffer is already clamped.
IREE_API_EXPORT iree_hal_buffer_t* iree_hal_buffer_view_buffer(
    const iree_hal_buffer_view_t* buffer_view);

// Returns the rank of the shape associated with the buffer view.
IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_shape_rank(const iree_hal_buffer_view_t* buffer_view);

// Returns a pointer to the shape dimensions; the array limit is defined by
// iree_hal_buffer_view_shape_rank.
IREE_API_EXPORT const iree_hal_dim_t* iree_hal_buffer_view_shape_dims(
    const iree_hal_buffer_view_t* buffer_view);

// Returns the value of the given dimension.
IREE_API_EXPORT iree_hal_dim_t iree_hal_buffer_view_shape_dim(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t index);

// Returns the dimensions of the shape in |out_shape| and its rank in
// |out_shape_rank|. |rank_capacity| indicates the number of dimensions
// available in the |out_shape| buffer. If there is not enough capacity to store
// all of the dimensions IREE_STATUS_OUT_OF_RANGE is returned.
// |out_shape_rank| can be omitted if the rank is already known.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_shape(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t rank_capacity,
    iree_hal_dim_t* out_shape, iree_host_size_t* out_shape_rank);

// Performs a **metadata update-only** reshape.
// The new rank and element count must match the existing values. The buffer
// contents are left untouched; if the buffer is not dense this may make the
// contents undefined.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_reshape(
    iree_hal_buffer_view_t* buffer_view, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank);

// Returns the total number of elements stored in the view.
IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_element_count(const iree_hal_buffer_view_t* buffer_view);

// Returns the element type of the buffer.
IREE_API_EXPORT iree_hal_element_type_t
iree_hal_buffer_view_element_type(const iree_hal_buffer_view_t* buffer_view);

// Returns the size of each element in the buffer view in bytes.
// Note that not all buffers are contiguous or densely packed.
IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_element_size(const iree_hal_buffer_view_t* buffer_view);

// Returns the encoding type of the buffer.
IREE_API_EXPORT iree_hal_encoding_type_t
iree_hal_buffer_view_encoding_type(const iree_hal_buffer_view_t* buffer_view);

// Returns the total size of the specified view in bytes.
// Note that not all buffers are contiguous or densely packed.
IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_view_byte_length(const iree_hal_buffer_view_t* buffer_view);

// Calculates a byte offset into the |buffer_view| at the given indices.
// Requires that the encoding and element type support indexing.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_compute_offset(
    const iree_hal_buffer_view_t* buffer_view, const iree_hal_dim_t* indices,
    iree_host_size_t indices_count, iree_device_size_t* out_offset);

// Calculates a byte range into the |buffer_view| of the given contiguous range.
// Requires that the encoding and element type support indexing.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_compute_range(
    const iree_hal_buffer_view_t* buffer_view,
    const iree_hal_dim_t* start_indices, iree_host_size_t indices_count,
    const iree_hal_dim_t* lengths, iree_host_size_t lengths_count,
    iree_device_size_t* out_start_offset, iree_device_size_t* out_length);

// Parses a serialized set of buffer elements in the canonical tensor format
// (the same as produced by iree_hal_buffer_view_format). The underlying buffer
// will be allocated with |buffer_allocator| as a host-local/device-visible
// buffer.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_parse(
    iree_string_view_t value, iree_hal_allocator_t* buffer_allocator,
    iree_hal_buffer_view_t** out_buffer_view);

// TODO(#5413): enum for printing mode (include shape, precision).

// Converts buffer view elements into a fully-specified string-form format like
// `2x4xi16=[[1 2][3 4]]`.
//
// |max_element_count| can be used to limit the total number of elements printed
// when the count may be large. Elided elements will be replaced with `...`.
//
// |buffer_capacity| defines the size of |buffer| in bytes and
// |out_buffer_length| will return the string length in characters. Returns
// IREE_STATUS_OUT_OF_RANGE if the buffer capacity is insufficient to hold the
// formatted elements and |out_buffer_length| will contain the required size.
//
// Follows the standard API string formatting rules. See iree/base/api.h.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_format(
    const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

// Prints buffer view elements into a fully-specified string-form format like
// `2x4xi16=[[1 2][3 4]]`.
//
// |max_element_count| can be used to limit the total number of elements printed
// when the count may be large. Elided elements will be replaced with `...`.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_fprint(
    FILE* file, const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t implementation details
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_buffer_view_destroy(
    iree_hal_buffer_view_t* buffer_view);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_BUFFER_VIEW_H_
