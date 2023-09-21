// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/parameter_provider.h"

IREE_API_EXPORT void iree_io_parameter_provider_retain(
    iree_io_parameter_provider_t* provider) {
  if (IREE_LIKELY(provider)) {
    iree_atomic_ref_count_inc(&provider->ref_count);
  }
}

IREE_API_EXPORT void iree_io_parameter_provider_release(
    iree_io_parameter_provider_t* provider) {
  if (IREE_LIKELY(provider) &&
      iree_atomic_ref_count_dec(&provider->ref_count) == 1) {
    provider->vtable->destroy(provider);
  }
}

IREE_API_EXPORT iree_status_t
iree_io_parameter_provider_notify(iree_io_parameter_provider_t* provider,
                                  iree_io_parameter_provider_signal_t signal) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE({
    switch (signal) {
      case IREE_IO_PARAMETER_PROVIDER_SIGNAL_RESUME:
        IREE_TRACE_ZONE_APPEND_TEXT(z0, "RESUME");
        break;
      case IREE_IO_PARAMETER_PROVIDER_SIGNAL_SUSPEND:
        IREE_TRACE_ZONE_APPEND_TEXT(z0, "SUSPEND");
        break;
      case IREE_IO_PARAMETER_PROVIDER_SIGNAL_LOW_MEMORY:
        IREE_TRACE_ZONE_APPEND_TEXT(z0, "LOW_MEMORY");
        break;
      default:
        IREE_TRACE_ZONE_APPEND_TEXT(z0, "(unknown)");
        break;
    }
  });
  iree_status_t status = provider->vtable->notify(provider, signal);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT bool iree_io_parameter_provider_query_support(
    iree_io_parameter_provider_t* provider, iree_string_view_t scope) {
  IREE_ASSERT_ARGUMENT(provider);
  return provider->vtable->query_support(provider, scope);
}

IREE_API_EXPORT iree_status_t iree_io_parameter_provider_load(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_string_view_t source_key,
    uint64_t source_offset, iree_hal_buffer_params_t target_params,
    iree_device_size_t length,
    iree_hal_buffer_t** IREE_RESTRICT out_target_buffer) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = provider->vtable->load(
      provider, device, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, source_scope, source_key, source_offset,
      target_params, length, out_target_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_provider_read(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_string_view_t source_key,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = provider->vtable->read(
      provider, device, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, source_scope, source_key, source_offset,
      target_buffer, target_offset, length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_provider_write(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_string_view_t target_scope, iree_string_view_t target_key,
    uint64_t target_offset, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = provider->vtable->write(
      provider, device, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, source_buffer, source_offset, target_scope,
      target_key, target_offset, length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_provider_gather(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_t* target_buffer,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);
  if (count == 0) {
    // Preserve the timeline when there's no work to do.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_device_queue_barrier(device, queue_affinity,
                                          wait_semaphore_list,
                                          signal_semaphore_list));
  } else if (count == 1) {
    // One span is just a read.
    iree_string_view_t key = iree_string_view_empty();
    iree_io_parameter_span_t span = {0};
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, enumerator.fn(enumerator.user_data, 0, &key, &span));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_parameter_provider_read(
                provider, device, queue_affinity, wait_semaphore_list,
                signal_semaphore_list, source_scope, key, span.parameter_offset,
                target_buffer, span.buffer_offset, span.length));
  } else {
    // Full multi-span gather.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, provider->vtable->gather(provider, device, queue_affinity,
                                     wait_semaphore_list, signal_semaphore_list,
                                     source_scope, target_buffer, count,
                                     enumerator));
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_parameter_provider_scatter(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_string_view_t target_scope,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);
  if (count == 0) {
    // Preserve the timeline when there's no work to do.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_device_queue_barrier(device, queue_affinity,
                                          wait_semaphore_list,
                                          signal_semaphore_list));
  } else if (count == 1) {
    // One span is just a write.
    iree_string_view_t key = iree_string_view_empty();
    iree_io_parameter_span_t span = {0};
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, enumerator.fn(enumerator.user_data, 0, &key, &span));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_parameter_provider_write(
                provider, device, queue_affinity, wait_semaphore_list,
                signal_semaphore_list, source_buffer, span.buffer_offset,
                target_scope, key, span.parameter_offset, span.length));
  } else {
    // Full multi-span scatter.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, provider->vtable->scatter(provider, device, queue_affinity,
                                      wait_semaphore_list,
                                      signal_semaphore_list, source_buffer,
                                      target_scope, count, enumerator));
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
