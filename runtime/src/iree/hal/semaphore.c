// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/semaphore.h"

#include <stddef.h>

#include "iree/hal/detail.h"
#include "iree/hal/device.h"

//===----------------------------------------------------------------------===//
// String utils
//===----------------------------------------------------------------------===//

static const iree_bitfield_string_mapping_t
    iree_hal_semaphore_compatibility_mappings[] = {
        {IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL, IREE_SVL("ALL")},
        {IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY, IREE_SVL("HOST_ONLY")},
        {IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_ONLY, IREE_SVL("DEVICE_ONLY")},
        {IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT, IREE_SVL("HOST_WAIT")},
        {IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT, IREE_SVL("DEVICE_WAIT")},
        {IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_SIGNAL, IREE_SVL("HOST_SIGNAL")},
        {IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_SIGNAL,
         IREE_SVL("DEVICE_SIGNAL")},
};

IREE_API_EXPORT iree_status_t iree_hal_semaphore_compatibility_parse(
    iree_string_view_t value, iree_hal_semaphore_compatibility_t* out_value) {
  return iree_bitfield_parse(
      value, IREE_ARRAYSIZE(iree_hal_semaphore_compatibility_mappings),
      iree_hal_semaphore_compatibility_mappings, out_value);
}

IREE_API_EXPORT iree_string_view_t iree_hal_semaphore_compatibility_format(
    iree_hal_semaphore_compatibility_t value,
    iree_bitfield_string_temp_t* out_temp) {
  return iree_bitfield_format_inline(
      value, IREE_ARRAYSIZE(iree_hal_semaphore_compatibility_mappings),
      iree_hal_semaphore_compatibility_mappings, out_temp);
}

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_t
//===----------------------------------------------------------------------===//

// Helper to get the typed vtable from a HAL semaphore.
static inline const iree_hal_semaphore_vtable_t* iree_hal_semaphore_vtable(
    iree_hal_semaphore_t* semaphore) {
  return (const iree_hal_semaphore_vtable_t*)((const iree_hal_resource_t*)
                                                  semaphore)
      ->vtable;
}

// Hand-written retain/release/destroy because the standard
// IREE_HAL_API_RETAIN_RELEASE macro reads ->destroy by member name, but
// with the embedded async vtable the destroy method is at ->async.destroy.
IREE_API_EXPORT void iree_hal_semaphore_destroy(
    iree_hal_semaphore_t* semaphore) {
  if (IREE_LIKELY(semaphore)) {
    iree_hal_semaphore_vtable(semaphore)->async.destroy(
        (iree_async_semaphore_t*)semaphore);
  }
}
IREE_API_EXPORT void iree_hal_semaphore_retain(
    iree_hal_semaphore_t* semaphore) {
  if (IREE_LIKELY(semaphore)) {
    iree_atomic_ref_count_inc(&((iree_hal_resource_t*)(semaphore))->ref_count);
  }
}
IREE_API_EXPORT void iree_hal_semaphore_release(
    iree_hal_semaphore_t* semaphore) {
  if (IREE_LIKELY(semaphore) &&
      iree_atomic_ref_count_dec(
          &((iree_hal_resource_t*)(semaphore))->ref_count) == 1) {
    iree_hal_semaphore_destroy(semaphore);
  }
}

IREE_API_EXPORT iree_status_t iree_hal_semaphore_create(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, initial_value);
  iree_status_t status =
      IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, create_semaphore)(
          device, queue_affinity, initial_value, flags, out_semaphore);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_semaphore_query(iree_hal_semaphore_t* semaphore, uint64_t* out_value) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  IREE_TRACE_ZONE_BEGIN(z0);
  // The async vtable query returns uint64_t directly with the failure sentinel
  // value indicating error. We adapt this to the HAL status + out_value API.
  uint64_t value = iree_hal_semaphore_vtable(semaphore)->async.query(
      (iree_async_semaphore_t*)semaphore);
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE)) {
    status = iree_hal_semaphore_failure_as_status(value);
    *out_value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  } else {
    *out_value = value;
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, *out_value);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_semaphore_signal(iree_hal_semaphore_t* semaphore, uint64_t new_value) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, new_value);
  iree_status_t status = iree_hal_semaphore_vtable(semaphore)->async.signal(
      (iree_async_semaphore_t*)semaphore, new_value, /*frontier=*/NULL);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_semaphore_fail(iree_hal_semaphore_t* semaphore,
                                             iree_status_t status) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, iree_status_code(status));
  iree_async_semaphore_fail((iree_async_semaphore_t*)semaphore, status);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t
iree_hal_semaphore_wait(iree_hal_semaphore_t* semaphore, uint64_t value,
                        iree_timeout_t timeout, iree_hal_wait_flags_t flags) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, value);
  iree_status_t status = iree_hal_semaphore_vtable(semaphore)->wait(
      semaphore, value, timeout, flags);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_semaphore_wait_source_ctl(
    iree_wait_source_t wait_source, iree_wait_source_command_t command,
    const void* params, void** inout_ptr) {
  iree_hal_semaphore_t* semaphore = (iree_hal_semaphore_t*)wait_source.self;
  const uint64_t target_value = wait_source.data;
  switch (command) {
    case IREE_WAIT_SOURCE_COMMAND_QUERY: {
      iree_status_code_t* out_wait_status_code = (iree_status_code_t*)inout_ptr;
      uint64_t current_value = 0;
      iree_status_t status =
          iree_hal_semaphore_query(semaphore, &current_value);
      if (!iree_status_is_ok(status)) {
        *out_wait_status_code = iree_status_code(status);
        iree_status_ignore(status);
      } else {
        *out_wait_status_code = current_value < target_value
                                    ? IREE_STATUS_DEFERRED
                                    : IREE_STATUS_OK;
      }
      return iree_ok_status();
    }
    case IREE_WAIT_SOURCE_COMMAND_WAIT_ONE: {
      const iree_timeout_t timeout =
          ((const iree_wait_source_wait_params_t*)params)->timeout;
      return iree_hal_semaphore_wait(semaphore, target_value, timeout,
                                     IREE_HAL_WAIT_FLAG_DEFAULT);
    }
    case IREE_WAIT_SOURCE_COMMAND_EXPORT: {
      const iree_wait_primitive_type_t target_type =
          ((const iree_wait_source_export_params_t*)params)->target_type;
      // TODO(benvanik): support exporting semaphores to real wait handles.
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
iree_hal_semaphore_await(iree_hal_semaphore_t* semaphore, uint64_t value) {
  IREE_ASSERT_ARGUMENT(semaphore);
  return (iree_wait_source_t){
      .self = semaphore,
      .data = value,
      .ctl = iree_hal_semaphore_wait_source_ctl,
  };
}

IREE_API_EXPORT iree_status_t iree_hal_semaphore_import_timepoint(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, value);
  iree_status_t status = iree_hal_semaphore_vtable(semaphore)->import_timepoint(
      semaphore, value, queue_affinity, external_timepoint);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_semaphore_export_timepoint(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, value);
  iree_status_t status = iree_hal_semaphore_vtable(semaphore)->export_timepoint(
      semaphore, value, queue_affinity, requested_type, requested_flags,
      out_external_timepoint);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_list_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_semaphore_list_retain(
    iree_hal_semaphore_list_t semaphore_list) {
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_semaphore_retain(semaphore_list.semaphores[i]);
  }
}

IREE_API_EXPORT void iree_hal_semaphore_list_release(
    iree_hal_semaphore_list_t semaphore_list) {
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_semaphore_release(semaphore_list.semaphores[i]);
  }
}

IREE_API_EXPORT iree_status_t iree_hal_semaphore_list_clone(
    const iree_hal_semaphore_list_t* source_list,
    iree_allocator_t host_allocator, iree_hal_semaphore_list_t* out_list) {
  IREE_ASSERT_ARGUMENT(out_list);
  *out_list = iree_hal_semaphore_list_empty();
  if (iree_hal_semaphore_list_is_empty(*source_list)) return iree_ok_status();

  // Single allocation for both arrays.
  iree_host_size_t semaphores_size =
      source_list->count * sizeof(iree_hal_semaphore_t*);
  iree_host_size_t values_size = source_list->count * sizeof(uint64_t);
  uint8_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, semaphores_size + values_size, (void**)&buffer));

  out_list->count = source_list->count;
  out_list->semaphores = (iree_hal_semaphore_t**)buffer;
  out_list->payload_values = (uint64_t*)(buffer + semaphores_size);
  for (iree_host_size_t i = 0; i < source_list->count; ++i) {
    out_list->semaphores[i] = source_list->semaphores[i];
    iree_hal_semaphore_retain(out_list->semaphores[i]);
    out_list->payload_values[i] = source_list->payload_values[i];
  }
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_semaphore_list_free(
    iree_hal_semaphore_list_t list, iree_allocator_t host_allocator) {
  iree_hal_semaphore_list_release(list);
  // Semaphores pointer is the base of the combined allocation.
  iree_allocator_free(host_allocator, list.semaphores);
}

IREE_API_EXPORT bool iree_hal_semaphore_list_poll(
    iree_hal_semaphore_list_t semaphore_list) {
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    // NOTE: this is unfortunately expensive in failure cases as it'll return
    // a clone (or maybe the original!) status. We rely on failures being
    // exceptional to make this acceptable.
    uint64_t current_value = 0;
    iree_status_t status =
        iree_hal_semaphore_query(semaphore_list.semaphores[i], &current_value);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return false;
    } else if (current_value < semaphore_list.payload_values[i]) {
      return false;  // not yet reached
    }
  }
  return true;
}

IREE_API_EXPORT iree_status_t
iree_hal_semaphore_list_signal(iree_hal_semaphore_list_t semaphore_list) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    status = iree_hal_semaphore_signal(semaphore_list.semaphores[i],
                                       semaphore_list.payload_values[i]);
    if (!iree_status_is_ok(status)) break;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_semaphore_list_fail(
    iree_hal_semaphore_list_t semaphore_list, iree_status_t signal_status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(signal_status)));

  // This handles cases of empty lists by dropping signal_status if not
  // consumed. Otherwise it clones the signal_status for each semaphore except
  // the last, which in the common case of a single timepoint fence means no
  // expensive clones.
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    const bool is_last = i == semaphore_list.count - 1;
    iree_status_t semaphore_status;
    if (is_last) {
      // Can transfer ownership of the signal status.
      semaphore_status = signal_status;
      signal_status = iree_ok_status();
    } else {
      // Clone status for this particular signal.
      semaphore_status = iree_status_clone(signal_status);
    }
    iree_hal_semaphore_fail(semaphore_list.semaphores[i], semaphore_status);
  }
  iree_status_ignore(signal_status);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_semaphore_list_wait(
    iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_hal_wait_flags_t flags) {
  if (!semaphore_list.count) return iree_ok_status();
  // HAL semaphores embed async semaphores at offset 0 (toll-free bridge).
  // The multi-wait handles all timepoint management internally.
  return iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL,
      (iree_async_semaphore_t**)semaphore_list.semaphores,
      semaphore_list.payload_values, semaphore_list.count, timeout,
      iree_allocator_system());
}

static void iree_hal_semaphore_list_swap_elements(
    iree_hal_semaphore_list_t* semaphore_list, iree_host_size_t i,
    iree_host_size_t j) {
  IREE_ASSERT(i >= 0 && i < semaphore_list->count);
  IREE_ASSERT(j >= 0 && j < semaphore_list->count);
  if (IREE_UNLIKELY(i == j)) {
    return;
  }

  iree_hal_semaphore_t* tmp_semaphore = semaphore_list->semaphores[i];
  uint64_t tmp_payload_value = semaphore_list->payload_values[i];

  semaphore_list->semaphores[i] = semaphore_list->semaphores[j];
  semaphore_list->payload_values[i] = semaphore_list->payload_values[j];

  semaphore_list->semaphores[j] = tmp_semaphore;
  semaphore_list->payload_values[j] = tmp_payload_value;
}

IREE_API_EXPORT void iree_hal_semaphore_list_erase(
    iree_hal_semaphore_list_t* semaphore_list, iree_host_size_t i) {
  IREE_ASSERT(semaphore_list->count > 0);
  iree_hal_semaphore_list_swap_elements(semaphore_list, i,
                                        semaphore_list->count - 1);
  --semaphore_list->count;
}
