// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/fence.h"

#include <stddef.h>

#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_fence_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_fence_create(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_hal_fence_t** out_fence) {
  IREE_ASSERT_ARGUMENT(out_fence);
  *out_fence = NULL;
  if (IREE_UNLIKELY(capacity >= UINT16_MAX)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "capacity %" PRIhsz " is too large for fence storage", capacity);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t semaphore_base = iree_host_align(
      sizeof(iree_hal_fence_t), iree_alignof(iree_hal_semaphore_t*));
  const iree_host_size_t value_base =
      iree_host_align(semaphore_base + capacity * sizeof(iree_hal_semaphore_t*),
                      iree_alignof(uint64_t));
  const iree_host_size_t total_size = value_base + capacity * sizeof(uint64_t);
  iree_hal_fence_t* fence = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&fence));
  iree_atomic_ref_count_init(&fence->ref_count);
  fence->host_allocator = host_allocator;
  fence->capacity = (uint16_t)capacity;
  fence->count = 0;

  *out_fence = fence;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_fence_create_at(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_allocator_t host_allocator, iree_hal_fence_t** out_fence) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_fence);
  *out_fence = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_fence_t* fence = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_fence_create(1, host_allocator, &fence));
  iree_status_t status = iree_hal_fence_insert(fence, semaphore, value);
  if (iree_status_is_ok(status)) {
    *out_fence = fence;
  } else {
    iree_hal_fence_release(fence);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// TODO(benvanik): actually join efficiently. Today we just create a fence that
// can hold the worst-case sum of all fence timepoints and then insert but it
// could be made much better. In most cases the joined fences have a near
// perfect overlap of semaphores and we are wasting memory.
IREE_API_EXPORT iree_status_t iree_hal_fence_join(
    iree_host_size_t fence_count, iree_hal_fence_t** fences,
    iree_allocator_t host_allocator, iree_hal_fence_t** out_fence) {
  IREE_ASSERT_ARGUMENT(out_fence);
  *out_fence = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Find the maximum required timepoint capacity.
  iree_host_size_t total_count = 0;
  for (iree_host_size_t i = 0; i < fence_count; ++i) {
    if (fences[i]) total_count += fences[i]->count;
  }

  // Empty list -> NULL.
  if (!total_count) {
    IREE_TRACE_ZONE_END(z0);
    return NULL;
  }

  // Create the fence with the maximum capacity.
  iree_hal_fence_t* fence = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_fence_create(total_count, host_allocator, &fence));

  // Insert all timepoints from all fences.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < fence_count; ++i) {
    iree_hal_semaphore_list_t source_list =
        iree_hal_fence_semaphore_list(fences[i]);
    for (iree_host_size_t j = 0; j < source_list.count; ++j) {
      status = iree_hal_fence_insert(fence, source_list.semaphores[j],
                                     source_list.payload_values[j]);
      if (!iree_status_is_ok(status)) break;
    }
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    *out_fence = fence;
  } else {
    iree_hal_fence_release(fence);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_fence_destroy(iree_hal_fence_t* fence) {
  IREE_ASSERT_ARGUMENT(fence);
  IREE_ASSERT_REF_COUNT_ZERO(&fence->ref_count);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = fence->host_allocator;

  iree_hal_semaphore_list_t list = iree_hal_fence_semaphore_list(fence);
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    iree_hal_semaphore_release(list.semaphores[i]);
  }
  iree_allocator_free(host_allocator, fence);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_fence_retain(iree_hal_fence_t* fence) {
  if (IREE_LIKELY(fence)) {
    iree_atomic_ref_count_inc(&fence->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_fence_release(iree_hal_fence_t* fence) {
  if (IREE_LIKELY(fence) && iree_atomic_ref_count_dec(&fence->ref_count) == 1) {
    iree_hal_fence_destroy(fence);
  }
}

IREE_API_EXPORT iree_hal_semaphore_list_t
iree_hal_fence_semaphore_list(iree_hal_fence_t* fence) {
  if (!fence) {
    return (iree_hal_semaphore_list_t){
        .count = 0,
        .semaphores = NULL,
        .payload_values = NULL,
    };
  }
  uint8_t* p = (uint8_t*)fence;
  const iree_host_size_t semaphore_base = iree_host_align(
      sizeof(iree_hal_fence_t), iree_alignof(iree_hal_semaphore_t*));
  const iree_host_size_t value_base = iree_host_align(
      semaphore_base + fence->capacity * sizeof(iree_hal_semaphore_t*),
      iree_alignof(uint64_t));
  return (iree_hal_semaphore_list_t){
      .count = fence->count,
      .semaphores = (iree_hal_semaphore_t**)(p + semaphore_base),
      .payload_values = (uint64_t*)(p + value_base),
  };
}

IREE_API_EXPORT iree_host_size_t
iree_hal_fence_timepoint_count(const iree_hal_fence_t* fence) {
  if (!fence) return 0;
  return fence->count;
}

IREE_API_EXPORT iree_status_t iree_hal_fence_insert(
    iree_hal_fence_t* fence, iree_hal_semaphore_t* semaphore, uint64_t value) {
  IREE_ASSERT_ARGUMENT(fence);
  IREE_ASSERT_ARGUMENT(semaphore);
  iree_hal_semaphore_list_t list = iree_hal_fence_semaphore_list(fence);

  // Try to find an existing entry with the same semaphore.
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    if (list.semaphores[i] == semaphore) {
      // Found existing; use max of both.
      list.payload_values[i] = iree_max(list.payload_values[i], value);
      return iree_ok_status();
    }
  }

  // Append to list if capacity remaining.
  if (list.count >= fence->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "fence unique semaphore capacity %u reached",
                            fence->capacity);
  }
  list.semaphores[list.count] = semaphore;
  iree_hal_semaphore_retain(semaphore);
  list.payload_values[list.count] = value;
  ++fence->count;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_fence_extend(
    iree_hal_fence_t* into_fence, iree_hal_fence_t* from_fence) {
  IREE_ASSERT_ARGUMENT(into_fence);
  IREE_ASSERT_ARGUMENT(from_fence);

  iree_hal_semaphore_list_t list = iree_hal_fence_semaphore_list(from_fence);
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_fence_insert(into_fence, list.semaphores[i],
                                               list.payload_values[i]));
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_fence_query(iree_hal_fence_t* fence) {
  if (!fence) return iree_ok_status();

  iree_hal_semaphore_list_t semaphore_list =
      iree_hal_fence_semaphore_list(fence);
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    uint64_t current_value = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_query(semaphore_list.semaphores[i], &current_value));
    if (current_value < semaphore_list.payload_values[i]) {
      return iree_status_from_code(IREE_STATUS_DEFERRED);
    }
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_fence_signal(iree_hal_fence_t* fence) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_hal_semaphore_list_signal(iree_hal_fence_semaphore_list(fence));
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_fence_fail(iree_hal_fence_t* fence,
                                         iree_status_t signal_status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_semaphore_list_fail(iree_hal_fence_semaphore_list(fence),
                               signal_status);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_fence_wait(iree_hal_fence_t* fence,
                                                  iree_timeout_t timeout) {
  if (!fence || !fence->count) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_semaphore_list_wait(
      iree_hal_fence_semaphore_list(fence), timeout);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_fence_wait_source_ctl(iree_wait_source_t wait_source,
                                             iree_wait_source_command_t command,
                                             const void* params,
                                             void** inout_ptr) {
  iree_hal_fence_t* fence = (iree_hal_fence_t*)wait_source.self;
  switch (command) {
    case IREE_WAIT_SOURCE_COMMAND_QUERY: {
      iree_status_code_t* out_wait_status_code = (iree_status_code_t*)inout_ptr;
      iree_status_t status = iree_hal_fence_query(fence);
      if (!iree_status_is_ok(status)) {
        *out_wait_status_code = iree_status_code(status);
        iree_status_ignore(status);
      } else {
        *out_wait_status_code = IREE_STATUS_OK;
      }
      return iree_ok_status();
    }
    case IREE_WAIT_SOURCE_COMMAND_WAIT_ONE: {
      const iree_timeout_t timeout =
          ((const iree_wait_source_wait_params_t*)params)->timeout;
      return iree_hal_fence_wait(fence, timeout);
    }
    case IREE_WAIT_SOURCE_COMMAND_EXPORT: {
      const iree_wait_primitive_type_t target_type =
          ((const iree_wait_source_export_params_t*)params)->target_type;
      // TODO(benvanik): support exporting fences to real wait handles.
      iree_wait_primitive_t* out_wait_primitive =
          (iree_wait_primitive_t*)inout_ptr;
      memset(out_wait_primitive, 0, sizeof(*out_wait_primitive));
      (void)target_type;
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "requested wait primitive type %d is unavailable",
                              (int)target_type);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplemented wait_source command");
  }
}

IREE_API_EXPORT iree_wait_source_t
iree_hal_fence_await(iree_hal_fence_t* fence) {
  if (!fence) return iree_wait_source_immediate();
  return (iree_wait_source_t){
      .self = fence,
      .data = 0,
      .ctl = iree_hal_fence_wait_source_ctl,
  };
}
