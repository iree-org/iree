// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_BUFFER_VIEW_UTIL_H_
#define IREE_HAL_BUFFER_VIEW_UTIL_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer_view.h"
#include "iree/hal/device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Buffer view math
//===----------------------------------------------------------------------===//

// Calculates the allocation size of a buffer view.
IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_size(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_device_size_t* out_allocation_size);

// Calculates a byte offset into a buffer at the given indices.
// Only works with densely-packed representations.
IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_offset(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_host_size_t indices_count,
    const iree_hal_dim_t* indices, iree_device_size_t* out_offset);

// Calculates a byte range into a buffer of the given contiguous range.
// Only works with densely-packed representations.
IREE_API_EXPORT iree_status_t iree_hal_buffer_compute_view_range(
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_host_size_t indices_count,
    const iree_hal_dim_t* start_indices, iree_host_size_t lengths_count,
    const iree_hal_dim_t* lengths, iree_device_size_t* out_start_offset,
    iree_device_size_t* out_length);

//===----------------------------------------------------------------------===//
// Buffer view allocation and generation
//===----------------------------------------------------------------------===//

// Allocates a copy of a buffer from |allocator| and wraps it in a buffer view.
//
// This is equivalent to:
//   1. iree_hal_buffer_compute_view_size
//   2. iree_hal_allocator_allocate_buffer
//   3. iree_hal_device_transfer_h2d
//   4. iree_hal_buffer_view_create
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_allocate_buffer_copy(
    iree_hal_device_t* device, iree_hal_allocator_t* allocator,
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_hal_buffer_params_t buffer_params, iree_const_byte_span_t initial_data,
    iree_hal_buffer_view_t** out_buffer_view);

typedef iree_status_t(IREE_API_PTR* iree_hal_buffer_view_generator_callback_t)(
    iree_hal_buffer_mapping_t* mapping, void* user_data);

// Generates a buffer view with its initial contents produced by a callback.
// When host and device memory are shared this allows direct generation into the
// target device buffer. If not shared this can avoid expensive transfer mapping
// operations at the cost of a transient host memory allocation. The mapped host
// pointer passed to the callback is only valid within the callback.
//
// Buffers allocated like this do not need the IREE_HAL_BUFFER_USAGE_MAPPING bit
// set; it will be added automatically if the allocator needs it and otherwise
// the memory can remain unmappable (and thus fully device isolated).
//
// As this _may_ require allocation of the entire buffer content in host memory
// it is always preferable to stage and issue copy commands via the device
// queue. Even better is to do all generation on-device via dispatches without
// the need to ever transfer. Usage of this method should be limited to times
// where device-side generation isn't possible or memory consumption is not a
// concern.
//
// This is equivalent to:
//   1. iree_hal_buffer_compute_view_size
//   2. iree_hal_allocator_allocate_buffer
//   3. iree_hal_buffer_map_range + callback + iree_hal_buffer_unmap_range
//   4. iree_hal_buffer_view_create
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_generate_buffer(
    iree_hal_device_t* device, iree_hal_allocator_t* allocator,
    iree_host_size_t shape_rank, const iree_hal_dim_t* shape,
    iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type,
    iree_hal_buffer_params_t buffer_params,
    iree_hal_buffer_view_generator_callback_t callback, void* user_data,
    iree_hal_buffer_view_t** out_buffer_view);

//===----------------------------------------------------------------------===//
// Buffer view parsing and printing
//===----------------------------------------------------------------------===//

// Parses a serialized set of buffer elements in the canonical tensor format
// (the same as produced by iree_hal_buffer_view_format). The underlying buffer
// will be allocated with |device_allocator| as a host-local/device-visible
// buffer.
IREE_API_EXPORT iree_status_t
iree_hal_buffer_view_parse(iree_string_view_t value, iree_hal_device_t* device,
                           iree_hal_allocator_t* device_allocator,
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
//
// |host_allocator| will be used for any transient allocations required while
// printing.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_fprint(
    FILE* file, const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_allocator_t host_allocator);

// Appends to |builder| a buffer view with contents without a trailing newline.
IREE_API_EXPORT iree_status_t iree_hal_buffer_view_append_to_builder(
    iree_hal_buffer_view_t* buffer_view, iree_host_size_t max_element_count,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_BUFFER_VIEW_UTIL_H_
