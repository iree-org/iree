// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_BUFFER_VIEW_H_
#define IREE_HAL_BUFFER_VIEW_H_

#include <stdbool.h>
#include <stdint.h>

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

enum iree_hal_numerical_type_e {
  IREE_HAL_NUMERICAL_TYPE_UNKNOWN = 0x00u,
  IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED = 0x01u,
  IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED = 0x02u,
  // TODO(benvanik): specialize with semantics from APFloat.
  IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE = 0x03u,
};
typedef uint8_t iree_hal_numerical_type_t;

#define IREE_HAL_ELEMENT_TYPE_VALUE(numerical_type, bit_count) \
  (((uint32_t)(numerical_type) << 24) | (uint32_t)(bit_count))

#define iree_hal_make_element_type(numerical_type, bit_count) \
  (iree_hal_element_type_t)(                                  \
      IREE_HAL_ELEMENT_TYPE_VALUE(numerical_type, bit_count))
#define iree_hal_element_numerical_type(element_type) \
  (iree_hal_numerical_type_t)((uint32_t)(element_type) >> 24)
#define iree_hal_element_bit_count(element_type) (size_t)((element_type)&0xFF)
#define iree_hal_element_byte_count(element_type) \
  ((iree_hal_element_bit_count(element_type) + 8 - 1) / 8)

// Defines the element type of a buffer in a standard format.
//
// Composed as a 32-bit bitfield to allow for opaque data types. Use
// iree_hal_make_element_type to make a bitfield with the appropriate ordering.
//
//   MSB ----------------------------------------------- LSB
//   [numerical type] [reserved] [reserved] [number of bits]
//
// clang-format off
enum iree_hal_element_type_e {
  IREE_HAL_ELEMENT_TYPE_NONE             = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,             0),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_8         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,             8),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_16        = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,            16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_32        = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,            32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_64        = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,            64),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_8           = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,      8),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_8           = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,    8),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_16          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,     16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_16          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,   16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_32          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,     32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_32          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,   32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_64          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,     64),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_64          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,   64),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_FLOAT_16         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE,         16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_FLOAT_32         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE,         32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_FLOAT_64         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE,         64),  // NOLINT
};
typedef uint32_t iree_hal_element_type_t;
// clang-format on

// A dimension within a shape.
typedef int32_t iree_hal_dim_t;

//===----------------------------------------------------------------------===//
// Buffer view math
//===----------------------------------------------------------------------===//

// Calculates the allocation size of a buffer view.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_compute_view_size(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type,
    iree_device_size_t* out_allocation_size);

// Calculates a byte offset into a buffer at the given indices.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_compute_view_offset(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type, const iree_hal_dim_t* indices,
    size_t indices_count, iree_device_size_t* out_offset);

// Calculates a byte range into a buffer of the given contiguous range.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_compute_view_range(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type, const iree_hal_dim_t* start_indices,
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
typedef struct iree_hal_buffer_view_s iree_hal_buffer_view_t;

// Creates a buffer view with the given |buffer|.
// |out_buffer_view| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_create(
    iree_hal_buffer_t* buffer, iree_hal_element_type_t element_type,
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_buffer_view_t** out_buffer_view);

// Creates a buffer view referencing a subview of the given |buffer_view|.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_subview(
    const iree_hal_buffer_view_t* buffer_view,
    const iree_hal_dim_t* start_indices, iree_host_size_t indices_count,
    const iree_hal_dim_t* lengths, iree_host_size_t lengths_count,
    iree_hal_buffer_view_t** out_buffer_view);

// Retains the given |buffer_view| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_buffer_view_retain(iree_hal_buffer_view_t* buffer_view);

// Releases the given |buffer_view| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_buffer_view_release(iree_hal_buffer_view_t* buffer_view);

// Returns the buffer underlying the buffer view.
// The caller must retain the returned buffer if they want to continue using it.
IREE_API_EXPORT iree_hal_buffer_t* IREE_API_CALL
iree_hal_buffer_view_buffer(const iree_hal_buffer_view_t* buffer_view);

// Returns the rank of the shape associated with the buffer view.
IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_hal_buffer_view_shape_rank(const iree_hal_buffer_view_t* buffer_view);

// Returns a pointer to the shape dimensions; the array limit is defined by
// iree_hal_buffer_view_shape_rank.
IREE_API_EXPORT const iree_hal_dim_t* IREE_API_CALL
iree_hal_buffer_view_shape_dims(const iree_hal_buffer_view_t* buffer_view);

// Returns the value of the given dimension.
IREE_API_EXPORT iree_hal_dim_t IREE_API_CALL iree_hal_buffer_view_shape_dim(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t index);

// Returns the dimensions of the shape in |out_shape| and its rank in
// |out_shape_rank|. |rank_capacity| indicates the number of dimensions
// available in the |out_shape| buffer. If there is not enough capacity to store
// all of the dimensions IREE_STATUS_OUT_OF_RANGE is returned.
// |out_shape_rank| can be omitted if the rank is already known.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_shape(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t rank_capacity,
    iree_hal_dim_t* out_shape, iree_host_size_t* out_shape_rank);

// Performs a **metadata update-only** reshape.
// The new rank and element count must match the existing values. The buffer
// contents are left untouched; if the buffer is not dense this may make the
// contents undefined.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_reshape(
    iree_hal_buffer_view_t* buffer_view, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank);

// Returns the total number of elements stored in the view.
IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_element_count(const iree_hal_buffer_view_t* buffer_view);

// Returns the element type of the buffer.
IREE_API_EXPORT iree_hal_element_type_t IREE_API_CALL
iree_hal_buffer_view_element_type(const iree_hal_buffer_view_t* buffer_view);

// Returns the size of each element in the buffer view in bytes.
// Note that not all buffers are contiguous or densely packed.
IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_hal_buffer_view_element_size(const iree_hal_buffer_view_t* buffer_view);

// Returns the total size of the specified view in bytes.
// Note that not all buffers are contiguous or densely packed.
IREE_API_EXPORT iree_device_size_t IREE_API_CALL
iree_hal_buffer_view_byte_length(const iree_hal_buffer_view_t* buffer_view);

// Calculates a byte offset into the |buffer_view| at the given indices.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_compute_offset(
    const iree_hal_buffer_view_t* buffer_view, const iree_hal_dim_t* indices,
    iree_host_size_t indices_count, iree_device_size_t* out_offset);

// Calculates a byte range into the |buffer_view| of the given contiguous range.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_compute_range(
    const iree_hal_buffer_view_t* buffer_view,
    const iree_hal_dim_t* start_indices, iree_host_size_t indices_count,
    const iree_hal_dim_t* lengths, iree_host_size_t lengths_count,
    iree_device_size_t* out_start_offset, iree_device_size_t* out_length);

// Parses a serialized set of buffer elements in the canonical tensor format
// (the same as produced by iree_hal_buffer_view_format). The underlying buffer
// will be allocated with |buffer_allocator| as a host-local/device-visible
// buffer.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_parse(
    iree_string_view_t value, iree_hal_allocator_t* buffer_allocator,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view);

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
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_format(
    const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t implementation details
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void IREE_API_CALL
iree_hal_buffer_view_destroy(iree_hal_buffer_view_t* buffer_view);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_BUFFER_VIEW_H_
