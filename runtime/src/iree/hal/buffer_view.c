// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/buffer_view.h"

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer_view_util.h"
#include "iree/hal/resource.h"

struct iree_hal_buffer_view_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;
  iree_hal_buffer_t* buffer;
  iree_hal_element_type_t element_type;
  iree_hal_encoding_type_t encoding_type;
  iree_device_size_t byte_length;
  iree_host_size_t shape_rank;
  iree_hal_dim_t shape[];
};

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_create(
    iree_hal_buffer_t* buffer, iree_host_size_t shape_rank,
    const iree_hal_dim_t* shape, iree_hal_element_type_t element_type,
    iree_hal_encoding_type_t encoding_type, iree_allocator_t host_allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer_view);

  *out_buffer_view = NULL;
  if (IREE_UNLIKELY(shape_rank > 0 && !shape)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no shape dimensions specified");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate and initialize the iree_hal_buffer_view_t struct.
  // Note that we have the dynamically-sized shape dimensions on the end.
  iree_hal_buffer_view_t* buffer_view = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator,
      sizeof(*buffer_view) + sizeof(iree_hal_dim_t) * shape_rank,
      (void**)&buffer_view);
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_init(&buffer_view->ref_count);
    buffer_view->host_allocator = host_allocator;
    buffer_view->buffer = buffer;
    iree_hal_buffer_retain(buffer_view->buffer);
    buffer_view->element_type = element_type;
    buffer_view->encoding_type = encoding_type;
    buffer_view->byte_length =
        iree_hal_element_dense_byte_count(buffer_view->element_type);
    buffer_view->shape_rank = shape_rank;
    for (iree_host_size_t i = 0; i < shape_rank; ++i) {
      buffer_view->shape[i] = shape[i];
      buffer_view->byte_length *= shape[i];
    }
    *out_buffer_view = buffer_view;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_create_like(
    iree_hal_buffer_t* buffer, iree_hal_buffer_view_t* like_view,
    iree_allocator_t host_allocator, iree_hal_buffer_view_t** out_buffer_view) {
  return iree_hal_buffer_view_create(
      buffer, iree_hal_buffer_view_shape_rank(like_view),
      iree_hal_buffer_view_shape_dims(like_view),
      iree_hal_buffer_view_element_type(like_view),
      iree_hal_buffer_view_encoding_type(like_view), host_allocator,
      out_buffer_view);
}

IREE_API_EXPORT void iree_hal_buffer_view_retain(
    iree_hal_buffer_view_t* buffer_view) {
  if (IREE_LIKELY(buffer_view)) {
    iree_atomic_ref_count_inc(&buffer_view->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_buffer_view_release(
    iree_hal_buffer_view_t* buffer_view) {
  if (IREE_LIKELY(buffer_view) &&
      iree_atomic_ref_count_dec(&buffer_view->ref_count) == 1) {
    iree_hal_buffer_view_destroy(buffer_view);
  }
}

IREE_API_EXPORT void iree_hal_buffer_view_destroy(
    iree_hal_buffer_view_t* buffer_view) {
  iree_allocator_t host_allocator = buffer_view->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_buffer_release(buffer_view->buffer);
  iree_allocator_free(host_allocator, buffer_view);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_hal_buffer_t* iree_hal_buffer_view_buffer(
    const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->buffer;
}

IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_shape_rank(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->shape_rank;
}

IREE_API_EXPORT const iree_hal_dim_t* iree_hal_buffer_view_shape_dims(
    const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->shape;
}

IREE_API_EXPORT iree_hal_dim_t iree_hal_buffer_view_shape_dim(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  if (IREE_UNLIKELY(index > buffer_view->shape_rank)) {
    return 0;
  }
  return buffer_view->shape[index];
}

IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_element_count(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  iree_host_size_t element_count = 1;
  for (iree_host_size_t i = 0; i < buffer_view->shape_rank; ++i) {
    element_count *= buffer_view->shape[i];
  }
  return element_count;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_shape(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t rank_capacity,
    iree_hal_dim_t* out_shape, iree_host_size_t* out_shape_rank) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  IREE_ASSERT_ARGUMENT(out_shape);
  if (out_shape_rank) {
    *out_shape_rank = 0;
  }

  if (out_shape_rank) {
    *out_shape_rank = buffer_view->shape_rank;
  }
  if (rank_capacity < buffer_view->shape_rank) {
    // Not an error; just a size query.
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  for (iree_host_size_t i = 0; i < buffer_view->shape_rank; ++i) {
    out_shape[i] = buffer_view->shape[i];
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_reshape(
    iree_hal_buffer_view_t* buffer_view, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  IREE_ASSERT_ARGUMENT(shape);

  if (shape_rank != buffer_view->shape_rank) {
    // Rank changes require reallocation of the structure as we inline the
    // shape dimensions. We could lighten this restriction to allow for rank
    // reduction but knowing that rank changes aren't allowed is easier than
    // remembering all the conditions in which they may be.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer view reshapes must have the same rank; "
                            "target=%zu, existing=%zu",
                            shape_rank, buffer_view->shape_rank);
  }

  iree_device_size_t new_element_count = 1;
  for (iree_host_size_t i = 0; i < shape_rank; ++i) {
    new_element_count *= shape[i];
  }
  iree_device_size_t old_element_count =
      iree_hal_buffer_view_element_count(buffer_view);
  if (new_element_count != old_element_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer view reshapes must have the same element "
                            "count; target=%" PRIdsz ", existing=%" PRIdsz,
                            new_element_count, old_element_count);
  }

  for (iree_host_size_t i = 0; i < shape_rank; ++i) {
    buffer_view->shape[i] = shape[i];
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_element_type_t
iree_hal_buffer_view_element_type(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->element_type;
}

IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_element_size(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return iree_hal_element_dense_byte_count(buffer_view->element_type);
}

IREE_API_EXPORT iree_hal_encoding_type_t
iree_hal_buffer_view_encoding_type(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->encoding_type;
}

IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_view_byte_length(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->byte_length;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_compute_offset(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t indices_count,
    const iree_hal_dim_t* indices, iree_device_size_t* out_offset) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return iree_hal_buffer_compute_view_offset(
      buffer_view->shape_rank, buffer_view->shape, buffer_view->element_type,
      buffer_view->encoding_type, indices_count, indices, out_offset);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_compute_range(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t indices_count,
    const iree_hal_dim_t* start_indices, iree_host_size_t lengths_count,
    const iree_hal_dim_t* lengths, iree_device_size_t* out_start_offset,
    iree_device_size_t* out_length) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return iree_hal_buffer_compute_view_range(
      buffer_view->shape_rank, buffer_view->shape, buffer_view->element_type,
      buffer_view->encoding_type, indices_count, start_indices, lengths_count,
      lengths, out_start_offset, out_length);
}
