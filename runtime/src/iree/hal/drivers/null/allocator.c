// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/null/allocator.h"

#include "iree/hal/drivers/null/buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_null_allocator_t
//===----------------------------------------------------------------------===//

// TODO(null): use one ID per address space or pool - each shows as a different
// track in tracing tools.
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_NULL_ALLOCATOR_ID = "{Null} unpooled";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_null_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_null_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_null_allocator_vtable;

static iree_hal_null_allocator_t* iree_hal_null_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_null_allocator_vtable);
  return (iree_hal_null_allocator_t*)base_value;
}

iree_status_t iree_hal_null_allocator_create(
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_allocator = NULL;

  iree_hal_null_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_null_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;

  // TODO(null): query device heaps, supported features (concurrent access/etc),
  // and prepare any pools that will be used during allocation. It's expected
  // that most failures that occur after creation are allocation
  // request-specific so preparing here will help keep the errors more
  // localized.
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "allocator not implemented");

  if (iree_status_is_ok(status)) {
    *out_allocator = (iree_hal_allocator_t*)allocator;
  } else {
    iree_hal_allocator_release((iree_hal_allocator_t*)allocator);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_null_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  iree_hal_null_allocator_t* allocator =
      iree_hal_null_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_null_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_null_allocator_t* allocator =
      (iree_hal_null_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_null_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_null_allocator_t* allocator =
      (iree_hal_null_allocator_t*)base_allocator;

  // TODO(null): if the allocator is retaining any unused resources they should
  // be dropped here. If the underlying implementation has pools or caches it
  // should be notified that a trim is requested. This is called in low-memory
  // situations or when IREE is not going to be used for awhile (low power modes
  // or suspension).
  (void)allocator;

  return iree_ok_status();
}

static void iree_hal_null_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_null_allocator_t* allocator =
        iree_hal_null_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
    // TODO(null): update statistics (merge).
  });
}

static iree_status_t iree_hal_null_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  iree_hal_null_allocator_t* allocator =
      iree_hal_null_allocator_cast(base_allocator);

  // TODO(null): return heap information. This is called at least once with a
  // capacity that may be 0 (indicating a query for the total count) and the
  // heaps should only be populated if capacity is sufficient to store all of
  // them.
  (void)allocator;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "heap query not implemented");

  return status;
}

static iree_hal_buffer_compatibility_t
iree_hal_null_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_null_allocator_t* allocator =
      iree_hal_null_allocator_cast(base_allocator);

  // TODO(null): set compatibility rules based on the implementation.
  // Note that the user may have requested that the allocator place the
  // allocation based on whatever is optimal for the indicated usage by
  // including the IREE_HAL_MEMORY_TYPE_OPTIMAL flag. It's still required that
  // the implementation meet all the requirements but it is free to place it in
  // either host or device memory so long as the appropriate bits are updated to
  // indicate where it landed.
  (void)allocator;
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_NONE;

  // We are now optimal.
  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;

  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (*allocation_size == 0) *allocation_size = 4;

  return compatibility;
}

static iree_status_t iree_hal_null_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_null_allocator_t* allocator =
      iree_hal_null_allocator_cast(base_allocator);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_null_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    // TODO(benvanik): make a helper for this.
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1, temp2;
    iree_string_view_t memory_type_str =
        iree_hal_memory_type_format(params->type, &temp0);
    iree_string_view_t usage_str =
        iree_hal_buffer_usage_format(params->usage, &temp1);
    iree_string_view_t compatibility_str =
        iree_hal_buffer_compatibility_format(compatibility, &temp2);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters; "
        "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
        (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
        usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
#endif  // IREE_STATUS_MODE
  }

  // TODO(null): allocate the underlying device memory. The impl_ptr is just
  // used for accounting and can be an opaque value (handle/etc) so long as it
  // is consistent between the alloc and free and unique to the buffer while it
  // is live. An example iree_hal_null_buffer_wrap is provided that can be used
  // for implementations that are managing memory using underlying allocators
  // and just wrapping those device pointers in the HAL buffer type. Other
  // implementations that require more tracking can provide their own buffer
  // types that do such tracking for them.
  (void)allocator;
  void* impl_ptr = NULL;
  (void)impl_ptr;
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "buffer allocation not implemented");

  if (iree_status_is_ok(status)) {
    // TODO(null): ensure this accounting is balanced in deallocate_buffer.
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_NULL_ALLOCATOR_ID, impl_ptr,
                           allocation_size);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static void iree_hal_null_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_null_allocator_t* allocator =
      iree_hal_null_allocator_cast(base_allocator);

  // TODO(null): free the underlying device memory here. Buffers allocated from
  // this allocator will call this method to handle cleanup. Note that because
  // this method is responsible for doing the base iree_hal_buffer_destroy and
  // the caller assumes the memory has been freed an implementation could pool
  // the buffer handle and return it in the future.
  (void)allocator;
  void* impl_ptr = NULL;
  (void)impl_ptr;

  // TODO(null): if the buffer was imported then this accounting may need to be
  // conditional depending on the implementation.
  bool was_imported = false;
  if (!was_imported) {
    IREE_TRACE_FREE_NAMED(IREE_HAL_NULL_ALLOCATOR_ID, impl_ptr);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
        &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
        iree_hal_buffer_allocation_size(base_buffer)));
  }

  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_null_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_null_allocator_t* allocator =
      iree_hal_null_allocator_cast(base_allocator);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  iree_device_size_t allocation_size = external_buffer->size;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_null_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)) {
    // TODO(benvanik): make a helper for this.
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1, temp2;
    iree_string_view_t memory_type_str =
        iree_hal_memory_type_format(params->type, &temp0);
    iree_string_view_t usage_str =
        iree_hal_buffer_usage_format(params->usage, &temp1);
    iree_string_view_t compatibility_str =
        iree_hal_buffer_compatibility_format(compatibility, &temp2);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot import a buffer with the given parameters; "
        "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
        (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
        usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot import a buffer with the given parameters");
#endif  // IREE_STATUS_MODE
  }

  // TODO(null): switch on external_buffer->type and import the buffer. See the
  // headers for more information on semantics. Most implementations can service
  // IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION by just wrapping the
  // underlying device pointer. Those that can service
  // IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION may be able to avoid a lot of
  // additional copies when moving data around between host and device or across
  // devices from different drivers.
  (void)allocator;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "external buffer type not supported");

  return status;
}

static iree_status_t iree_hal_null_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  iree_hal_null_allocator_t* allocator =
      iree_hal_null_allocator_cast(base_allocator);

  // TODO(null): switch on requested_type and export as appropriate. Most
  // implementations can service IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION
  // by just exposing the underlying device pointer. Those that can service
  // IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION may be able to avoid a lot of
  // additional copies when moving data around between host and device or across
  // devices from different drivers.
  (void)allocator;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "external buffer type not supported");
}

static const iree_hal_allocator_vtable_t iree_hal_null_allocator_vtable = {
    .destroy = iree_hal_null_allocator_destroy,
    .host_allocator = iree_hal_null_allocator_host_allocator,
    .trim = iree_hal_null_allocator_trim,
    .query_statistics = iree_hal_null_allocator_query_statistics,
    .query_memory_heaps = iree_hal_null_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_null_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_null_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_null_allocator_deallocate_buffer,
    .import_buffer = iree_hal_null_allocator_import_buffer,
    .export_buffer = iree_hal_null_allocator_export_buffer,
};
